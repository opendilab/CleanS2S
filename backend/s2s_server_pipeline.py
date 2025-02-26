from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import os
import re
import json
import base64
import random
import time
import threading
from queue import Queue, Empty
from threading import Event, Thread
from time import perf_counter
import glob
import requests
from io import BytesIO

import numpy as np
import torch
import nltk
import asyncio
import websockets
from rich.console import Console
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
# ASR
from funasr import AutoModel
# TTS
import torchaudio
# try:
#     from cosyvoice.utils.file_utils import load_wav
#     from cosyvoice.cli.cosyvoice import CosyVoice
# except ImportError as e:
#     print(f"Failed to import CosyVoice modules. Please ensure cosyvoice is installed (pip install cosyvoice). Error: {e}")
#     load_wav = None
#     CosyVoice = None

# Ensure that the necessary NLTK resources are available
# try:
#     nltk.data.find('tokenizers/punkt_tab')
#     nltk.data.find('averaged_perceptron_tagger_eng')
# except LookupError:
#     nltk.download('punkt_tab')
#     nltk.download('averaged_perceptron_tagger_eng')

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = os.path.dirname(__file__)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "compile_tmp")
# torch._inductor.config.fx_graph_cache = True
# # mind about this parameter ! should be >= 2 * number of padded prompt sizes for TTS
# torch._dynamo.config.cache_size_limit = 15

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)

logger.info(f'BEGIN LOGGER {__name__}')
console = Console()
global pipeline_start
pipeline_start = None

CLEANS2S_SMART_INTERACTION_PROMPT = """
# 角色
你是一个说话简洁但风趣幽默的人，擅长与他人轻松愉快地交流。

# 能力
1. 风格适应：能根据用户的对话风格，灵活调整自己的说话方式和口语化程度，让对话更加自然亲切。
2. 理解纠错：如果用户输入中有难以理解的字和词汇，可能是语音识别的误差，你会尝试给出几种可能的解释，并询问用户以确认，确定信息后再开始后续回答。如果没有，正常回答即可。
3. 避免使用复杂的格式（比如 markdown 或强调符号**等），简单断句或分段即可，分点列举信息时不要超过4种，保持对话的流畅性。
4. 分段回答：对于较长的知识性回答，不要直接长篇大论说很多，你需要先给出总体介绍，然后分成多个部分（不超过4个），每个部分只用简短的关键词指代，不要过度展开，用户如果追问细节，你再给出相应的详细介绍。

# 例子
用户输入1
"我今天吃了个超级大的披萨，结果肚子疼得要命，是不是因为披萨泰拉了？"

智能体输出1
"哎呀，心疼你一秒！肚子疼得要命，你是不是在说披萨太辣了？还是你觉得可能是其他原因呢？比如，披萨是不是品质太差太垃圾了？我们一起想想办法，看看怎么样能帮你缓解一下？"

用户输入2
"你会和面吗？能给我讲讲做法吗？"

智能体输出2
和面的做法其实很简单，整体大概可以分为四步：准备材料、混合搅拌、揉面、醒面。你具体想了解哪一块呢？
"""
CLEANS2S_SMART_POST_QUESTION_PROMPT = """
对上文进行分析，判断语境是否生成联想问题，如果是，请生成基于上文信息，用户可能还会问的3个问题，格式规范为：【用户可能想问】：<问题1>|<问题2>|<问题3>。如果不是，输出"空"。
"""
CLEANS2S_SMART_POST_QUESTION_PATTERN = r"【用户可能想问】：([^|]+)\|([^|]+)\|([^|]+)"


def get_readable_time(timestamp: float) -> str:
    dt_object = datetime.fromtimestamp(timestamp)
    milliseconds = dt_object.microsecond // 1000
    readable_time = dt_object.strftime("%Y-%m-%d-%H-%M-%S") + f"-{milliseconds:03}"
    return readable_time


def save_fn(filename: str, data: Any) -> None:
    """Data type: numpy.array, str, int, float, None"""
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np_data_name = f"{filename.split('.')[0]}_audio.npy"
            np.save(np_data_name, value)
            data[key] = np_data_name
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers: List['BaseHandler']) -> None:
        """
        Arguments:
            - handlers (List[BaseHandler]): List of handlers to be executed in separate threads. Such as STT (ASR), \
                LLM, TTS, and websocket receiver and sender.
        """
        self.handlers = handlers
        self.threads = []

    def start(self) -> None:
        """
        Start all threads and store them in the threads list.
        """
        for handler in self.handlers:
            if isinstance(handler, BaseHandler):
                thread = threading.Thread(target=handler.run)
            else:
                # handlers with asyncio, which should start as a coroutine with individual event loop
                thread = threading.Thread(target=lambda: asyncio.run(handler.start()))
            self.threads.append(thread)
            thread.start()

    def stop(self) -> None:
        """
        Stop all threads and related stop events.
        """
        for handler in self.handlers:
            handler.stop_event.set()
        for thread in self.threads:
            thread.join()

    def join(self) -> None:
        """
        Join all threads.
        """
        for thread in self.threads:
            thread.join()


class BaseHandler(ABC):
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    To stop a handler properly, set the stop_event and.
    To restart a handler from current connections, set the cur_conn_end_event.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed
    in the output queue. The cleanup method handles stopping the handler.
    """

    def __init__(self, stop_event: Event, cur_conn_end_event: Event, queue_in: Queue, queue_out: Queue) -> None:
        """
        Arguments:
            - stop_event (Event): Event used to stop the processor.
            - cur_conn_end_event (Event): Event used to indicate whether the current connection has ended.
            - queue_in (Queue): Input queue.
            - queue_out (Queue): Output queue.
        """
        self.stop_event = stop_event
        self.cur_conn_end_event = cur_conn_end_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self._times = []

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def run(self) -> None:
        """
        Runs the main loop of the processor.
        Gets objects from the input queue, calls the process method on them, and puts the results into the output queue.
        Stops when stop_event is set.
        """
        while not self.stop_event.is_set():
            try:
                input = self.queue_in.get(timeout=0.2)
            except Empty:
                # It queue_in is empty, check if the current connection has ended.
                # If so, clear the current state and wait for the next connection.
                if self.cur_conn_end_event.is_set():
                    self.clear_current_state()
                    time.sleep(0.1)
                continue
            start_time = perf_counter()
            for output in self.process(input):
                self._times.append(perf_counter() - start_time)
                logger.debug(f"{self.__class__.__name__}: {self.last_time: .3f} s")
                self.queue_out.put(output)
                start_time = perf_counter()

        self.cleanup()

    @property
    def last_time(self) -> float:
        """
        Get the time of the last processing.
        Returns:
            - last_time (float): The time of the last processing.
        """
        return self._times[-1]

    def cleanup(self) -> None:
        """
        Cleanup related resources and stop the handler.
        """
        pass

    def clear_current_state(self) -> None:
        """
        Clear the current state when the current connection ends. Such as flush the input and output queues.
        """
        while not self.queue_in.empty():
            self.queue_in.get()
            time.sleep(0.01)
        while not self.queue_out.empty():
            self.queue_out.get()
            time.sleep(0.01)


class VADIterator:
    """
    Voice Activity Detection (VAD) iterator.
    Mainly taken from https://github.com/snakers4/silero-vad
    Class for stream imitation
    """

    def __init__(
            self,
            model: torch.nn.Module,
            threshold: float = 0.5,
            sampling_rate: int = 16000,
            min_silence_duration_ms: int = 100,
            speech_pad_ms: int = 30,
            max_speech_ms: int = 60000,
    ) -> None:
        """
        Arguments:
            - model (torch.nn.Module): Preloaded .jit/.onnx silero VAD model.
            - threshold (float): Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, \
                probabilities ABOVE this value are considered as SPEECH. It is better to tune this parameter for each \
                dataset separately, but "lazy" 0.5 is pretty good for most datasets.
            - sampling_rate (int): Sampling rate of the audio. Currently, silero VAD models support 8000 and 16000.
            - min_silence_duration_ms (int): In the end of each speech chunk, wait for min_silence_duration_ms before \
                separating it.
            - speech_pad_ms (int): Final speech chunks are padded by speech_pad_ms each side.
            - max_speech_ms (int): Maximum speech duration in milliseconds. If the speech duration exceeds this value, \
                the speech chunk will be separated (return self.buffer).
        """

        self.model = model
        self.threshold = threshold
        assert self.threshold >= 0.15, "Threshold should be >= 0.15"
        self.sampling_rate = sampling_rate
        self.is_speaking = False
        self.buffer = []

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.max_speech_samples = sampling_rate * max_speech_ms / 1000
        self.reset_states()

    def reset_states(self) -> None:
        """
        Reset the state of the VAD iterator.
        """
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process the audio chunk and detect voice activity. If voice activity is detected, the audio data will be
        accumulated until the end of speech is detected. If not, None will be returned.
        Arguments:
            - x (torch.Tensor): Audio chunk data.
        Returns:
            - spoken_utterance (Optional[torch.Tensor]): The detected speech data.
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:  # noqa
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            return None

        # check if speech is end (detect enough slience chunk)
        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                # end of speak
                self.temp_end = 0
                self.triggered = False
                spoken_utterance = self.buffer
                self.buffer = []
                return spoken_utterance
        # check if speech is end (excess max_speech_samples)
        if len(self.buffer) * window_size_samples >= self.max_speech_samples:
            # end of speak
            self.temp_end = 0
            self.triggered = False
            spoken_utterance = self.buffer
            self.buffer = []
            return spoken_utterance

        if self.triggered:
            self.buffer.append(x)

        return None


class SocketVADReceiver:
    """
    Handles reception of the audio packets from the client and voice activity detectionself.
    When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed to
    the following part (ASR/STT).
    SocketVADReceiver is a asyncio handler.
    """

    def __init__(
            self,
            stop_event: Event,
            cur_conn_end_event: Event,
            queue_out: Queue,
            should_listen: Event,
            interruption_event: Event,
            host: str = '0.0.0.0',
            port: int = 9001,
            chunk_size: int = 2048,
            enable_interruption: bool = False,
            thresh: float = 0.3,
            sample_rate: int = 16000,
            min_silence_ms: int = 1200,
            min_speech_ms: int = 400,
            max_speech_ms: float = float('inf'),
            speech_pad_ms: int = 30,
    ) -> None:
        """
        Arguments:
            - stop_event (Event): Event used to stop the receiver.
            - cur_conn_end_event (Event): Event used to indicate whether the current connection has ended.
            - queue_out (Queue): Output queue.
            - should_listen (Event): Event used to indicate whether the receiver should listen to the audio data.
            - interruption_event (Event): Event used to trigger user interruption.
            - host (str): Host address.
            - port (int): Port number.
            - chunk_size (int): Size of the audio chunk.
            - enable_interruption (bool): Whether to enable user interruption.
            - thresh (float): Speech threshold.
            - sample_rate (int): Sampling rate of the audio.
            - min_silence_ms (int): Minimum silence duration in milliseconds.
            - min_speech_ms (int): Minimum speech duration in milliseconds.
            - max_speech_ms (float): Maximum speech duration in milliseconds.
            - speech_pad_ms (int): Speech padding in milliseconds.
        """
        self.stop_event = stop_event
        self.cur_conn_end_event = cur_conn_end_event
        self.queue_out = queue_out
        self.should_listen = should_listen
        self.interruption_event = interruption_event
        self.chunk_size = chunk_size
        self.host = host
        self.port = port
        self.server = None
        self.restart_event = asyncio.Event()

        self.enable_interruption = enable_interruption
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.frontend_is_playing = False
        # user_input_count is used to count the number of user inputs, which is used to give the unique id to the
        # input and output data
        self.user_input_count = 0

    async def handler(self, ws: websockets, path: str) -> None:
        """
        The processing function of receiving the WebSocket request.
        Set the current connection end event when the connection is closed to notify other components to restart.
        """
        logger.info("Receiver WebSocket connected")
        self.should_listen.set()
        while not self.stop_event.is_set():
            try:
                # res is naive string or json string
                res = await ws.recv()

                # only for forward server: naive string
                if res.startswith("ConnectionClosedError"):
                    logger.error(
                        f"(forward)Receiver WebSocket conn error closed. Should listen {self.should_listen.is_set()}"
                    )
                    await ws.send(json.dumps({"placeholder": ""}))
                    self.cur_conn_end_event.set()
                    self.restart()
                    break
                # only for forward server
                elif res.startswith("ConnectionClosedOK"):
                    logger.info(
                        f"(forward)Receiver WebSocket connection OK closed. Should listen {self.should_listen.is_set()}"
                    )
                    await ws.send(json.dumps({"placeholder": ""}))
                    self.user_input_count = 0
                    self.cur_conn_end_event.set()
                    continue

                # normal case: json string
                json_data = json.loads(res)
                uid = json_data["uid"]
                # is_playing is a indication of the frontend playing status.
                is_playing = json_data.get("is_playing", "placeholder")
                if is_playing in ['true', 'false']:
                    # logger.info(f'set frontend_is_playing: {res}')
                    self.frontend_is_playing = (is_playing == 'true')

                text = json_data.get("text")
                audio = json_data.get("audio")

                # user directly send text question
                if text is not None:
                    # when res equals 'new topic', it means that we should clear the chat history.
                    if text == 'new topic':
                        await ws.send(json.dumps({"placeholder": ""}))
                        self.user_input_count = 0
                        self.cur_conn_end_event.set()
                    else:
                        self.user_input_count += 1
                        self.queue_out.put({"data": text, "user_input_count": self.user_input_count, "uid": uid})
                        # If text string is detected and frontend is playing, trigger the user interruption
                        if self.frontend_is_playing:
                            self.interruption_event.set()
                            logger.info('Trigger interruption')
                        await ws.send(json.dumps({"placeholder": ""}))
                elif audio is not None:
                    vad = False
                    audio = base64.b64decode(audio)
                    if self.should_listen.is_set():
                        for i in range(len(audio) // self.chunk_size):
                            data = audio[i * self.chunk_size:(i + 1) * self.chunk_size]
                            vad_result = self.vad(data)
                            if vad_result is not None:
                                self.user_input_count += 1
                                self.queue_out.put(
                                    {
                                        "data": vad_result,
                                        "user_input_count": self.user_input_count,
                                        "uid": uid
                                    }
                                )
                                # If VAD is detected and frontend is playing, trigger the user interruption
                                if self.frontend_is_playing:
                                    self.interruption_event.set()
                                    logger.info('Trigger interruption')
                                await ws.send(json.dumps({"return_info": "VAD detected"}))
                                vad = True
                    if not vad:
                        await ws.send(json.dumps({"placeholder": ""}))
            except websockets.ConnectionClosedError as e:
                logger.error(f"Receiver WebSocket connection error closed: {e}")
                self.cur_conn_end_event.set()
                self.restart()
                break
            except websockets.ConnectionClosedOK:
                logger.info("Receiver WebSocket connection OK closed")
                self.user_input_count = 0
                self.cur_conn_end_event.set()
                break
        logger.info("Receiver WebSocket connection end")

    async def run(self) -> None:
        """
        Runs a WebSocket receiver server, waiting for clients to connect and send audio data.
        """
        while not self.stop_event.is_set():
            try:
                # set ping_interval and ping_timeout to None to disable ping-pong, which makes it more stable
                self.server = await websockets.serve(
                    self.handler, self.host, self.port, ping_interval=None, ping_timeout=None
                )
                logger.info(f'Receiver WebSocket server started at {self.host}:{self.port}')
                await self.restart_event.wait()
                self.restart_event.clear()
            except websockets.exceptions.InvalidState:
                logger.error("Receiver WebSocket server encountered an invalid state, restarting...")
            finally:
                if self.server:
                    self.server.close()
                    await self.server.wait_closed()
                    self.server = None
                await asyncio.sleep(0.1)  # Wait before attempting to restart

    def start(self) -> None:
        """
        Start the WebSocket server and create new individual event loop.
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.run())
        except KeyboardInterrupt:
            logger.info("Receiver WebSocket server stopped by user")
        finally:
            loop.run_until_complete(self.stop())

    async def stop(self) -> None:
        """
        Stop the WebSocket server and clear related resources.
        """
        self.stop_event.set()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("Receiver WebSocket server closed")

    def restart(self) -> None:
        """
        Restart the WebSocket server for new connections. Set restart event and reset state variables.
        """
        self.restart_event.set()
        self.frontend_is_playing = False
        self.user_input_count = 0

    def vad(self, audio_chunk: bytes) -> Optional[np.ndarray]:
        """
        Use the VAD model to process the audio chunk and detect voice activity.
        Arguments:
            - audio_chunk (bytes): Audio chunk data.
        Returns:
            - vad_result (Optional[np.ndarray]): The detected speech data in numpy array.
        """
        audio_float32 = np.frombuffer(audio_chunk, dtype=np.float32)
        vad_output = self.iterator(torch.from_numpy(audio_float32))
        # if vad_output is not None:
        #     print(f'vad {len(vad_output)}', audio_float32.shape)
        # else:
        #     print('vad None', audio_float32.shape)
        if vad_output is not None and len(vad_output) != 0:
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            logger.info(f"VAD: end of speech detected: {duration_ms}")
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.info(f"audio input of duration: {len(array) / self.sample_rate}s, skipping")
            else:
                # If not enable interruption, disable should listen after VAD detected, and enable it after TTS
                if not self.enable_interruption:
                    self.should_listen.clear()
                    logger.info("Disable should listen")
                return array


class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for encoding bytes to base64 strings when serializing in SocketSender.
    """

    def default(self, obj: Any) -> Union[str, Any]:
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        return super().default(obj)


class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    SocketSender is a asyncio handler.
    """

    def __init__(
            self,
            stop_event: Event,
            cur_conn_end_event: Event,
            queue_in: Queue,
            host: str = '0.0.0.0',
            port: int = 9002
    ) -> None:
        """
        Arguments:
            - stop_event (Event): Event used to stop the sender.
            - cur_conn_end_event (Event): Event used to indicate whether the current connection has ended.
            - queue_in (Queue): Input queue.
            - host (str): Host address.
            - port (int): Port number.
        """
        self.stop_event = stop_event
        self.cur_conn_end_event = cur_conn_end_event
        self.queue_in = queue_in
        self.host = host
        self.port = port
        self.server = None
        self.restart_event = asyncio.Event()

    async def handler(self, websocket: websockets, path: str) -> None:
        """
        The processing function of the WebSocket request.
        """
        logger.info("Sender WebSocket connected")
        while not self.cur_conn_end_event.is_set():
            try:
                data = self.queue_in.get(timeout=3)
            except Empty:  # The timeout and the try-except block are used to check if the current connection has ended.
                continue
            # answer_audio: np.ndarray, dtype: np.int16
            data['answer_audio'] = bytes(data['answer_audio'])

            try:
                # use CustomEncoder to encode bytes to base64 strings
                await websocket.send(json.dumps(data, cls=CustomEncoder))
            except websockets.ConnectionClosedError as e:
                logger.error(f"Sender WebSocket connection error closed: {e}")
                self.restart()
                break
            except websockets.ConnectionClosedOK:
                logger.info("Sender WebSocket connection OK closed")
                break
        # If the current connection has ended, clear the current state and wait for the next connection.
        # Usually, this event is triggered by the receiver.
        if self.cur_conn_end_event.is_set():
            self.restart()
        logger.info("Sender WebSocket connection end")

    async def run(self) -> None:
        """
        Runs a WebSocket sender server, waiting for clients to connect. The sender will poll the input queue and send
        the generated audio data to the clients.
        """
        while not self.stop_event.is_set():
            try:
                self.cur_conn_end_event.clear()
                # set ping_interval and ping_timeout to None to disable ping-pong, which makes it more stable
                self.server = await websockets.serve(
                    self.handler, self.host, self.port, ping_interval=None, ping_timeout=None
                )
                logger.info(f'Sender WebSocket server started at {self.host}:{self.port}')
                await self.restart_event.wait()
                self.restart_event.clear()
            except websockets.exceptions.InvalidState:
                logger.error("Sender WebSocket server encountered an invalid state, restarting...")
            finally:
                if self.server:
                    self.server.close()
                    await self.server.wait_closed()
                    self.server = None
                await asyncio.sleep(0.1)  # Wait before attempting to restart

    def start(self) -> None:
        """
        Start the WebSocket server and create new individual event loop.
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.run())
        except KeyboardInterrupt:
            logger.info("Sender WebSocket server stopped by user")
        finally:
            loop.run_until_complete(self.stop())

    async def stop(self) -> None:
        """
        Stop the WebSocket server and clear related resources.
        """
        self.stop_event.set()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("Sender WebSocket server closed")

    def restart(self) -> None:
        """
        Restart the WebSocket server for new connections. Set restart event.
        """
        self.restart_event.set()


class ParaFormerSTTHandler(BaseHandler):
    """
    Use the Paraformer model to convert the input audio data to text and speech recognition (arXiv:2206.08317).
    This class provides methods to initialize the model, warm up the model, and process audio data.
    Through these methods, speech-to-text conversion can be achieved efficiently.
    """

    def __init__(
            self,
            stop_event: Event,
            cur_conn_end_event: Event,
            queue_in: Queue,
            queue_out: Queue,
            model_name: str,
            device: str = "cuda",
            dtype: str = "float16",
            compile: Optional[bool] = True,
            save_data: Optional[bool] = False,
    ) -> None:
        """
        Arguments:
            - stop_event (Event): Event used to stop the processor.
            - cur_conn_end_event (Event): Event used to indicate whether the current connection has ended.
            - queue_in (Queue): Input queue.
            - queue_out (Queue): Output queue.
            - model_name (str): The name of the model to use.
            - device (str): The device to use for processing.
            - dtype (str): The data type to use for processing.
            - compile (Optional[bool]): Whether to use `torch.compile` to speed up the model.
            - save_data (Optional[bool]): Whether to save processed data into `s2d_data` directory.
        """
        super().__init__(stop_event, cur_conn_end_event, queue_in, queue_out)
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.compile = compile
        self.save_data = save_data

        prefix = model_name
        # vad model and punc model are necessary for the Paraformer model
        self.model = AutoModel(
            model=f"{prefix}/seaco_paraformer",
            punc_model=f"{prefix}/punc_ct",
            vad_model=f"{prefix}/fsmn_vad",
            vad_kwargs={
                "max_single_segment_time": 60000,
                "window_size_ms": 200
            },
            ncpu=8,
            device=self.device,
            sentence_timestamp=True,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
            # spk_mode="punc_segment"
        )

        # compile
        if self.compile:
            self.model.model = torch.compile(self.model.model, mode="default", fullgraph=True)
            print(self.model.model)
        self.warmup()

    def warmup(self) -> None:
        """
        Model warm-up to improve the model's responsiveness and performance in real-world use.
        """
        logger.info(f"Warming up {self.__class__.__name__}")

        n_steps = 10
        dummy_input = torch.randn((160000, ))  # to device will be implemented inside paraformer

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, batch_size_s=300, batch_size_threshold_s=60)

        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, inputs: Dict[str, Union[np.ndarray, str, int]]) -> Dict[str, Union[str, int, bool]]:
        """
        Process the input acquired from queue_in (from SocketVADReceiver) and generate the ASR output with Paraformer.
        Arguments:
            - inputs (Dict[str, Union[np.ndarray, str, int]]): The input data acquired from queue_in. The data \
                contains the np.ndarray format audio data, string user id (uid) and integer user input count.
        Returns (Yield):
            - output (Dict[str, Union[str, int, bool]]): The output data containing the ASR result, user id, bool flag \
                about audio/text input and user input count.
        """
        logger.info("inference ASR paraformer...")
        spoken_prompt, user_input_count, uid = inputs["data"], inputs["user_input_count"], inputs["uid"]

        global pipeline_start
        pipeline_start = perf_counter()

        # user directly send text question
        if isinstance(spoken_prompt, str):
            console.print(f"[yellow]{time.ctime()}\tUSER: {spoken_prompt}")
            if self.save_data:
                save_fn(
                    f's2s_data/{uid}_{user_input_count}_input.json', {
                        "audio": None,
                        "text": spoken_prompt,
                        "audio_input": False,
                        "time": time.ctime()
                    }
                )
            yield {"data": spoken_prompt, "user_input_count": user_input_count, "uid": uid, "audio_input": False}
        else:
            res = self.model.generate(input=spoken_prompt, batch_size_s=300, batch_size_threshold_s=60)
            try:
                pred_text = "".join([t["text"] for t in res[0]["sentence_info"]])
            except Exception:
                # If the ASR model fails to generate the text, use the default content.
                logger.error("ASR error, use default content")
                pred_text = "然后，"

            logger.info("finish ASR paraformer inference")
            console.print(f"[yellow]{time.ctime()}\tUSER: {pred_text}")
            if self.save_data:
                save_fn(
                    f's2s_data/{uid}_{user_input_count}_input.json', {
                        "audio": spoken_prompt,
                        "text": pred_text,
                        "audio_input": True,
                        "time": time.ctime()
                    }
                )

            yield {"data": pred_text, "user_input_count": user_input_count, "uid": uid, "audio_input": True}


class Chat:
    """
    Handles the chat using to control the conversation flow and generate the prompt for the language model.
    """

    def __init__(self, size: int, user_role: str = "user", assistant_role: str = "assistant") -> None:
        """
        Arguments:
            - size (int): The size of the chat cache (history).
            - user_role (str): The role of the user in the conversation.
            - assistant_role (str): The role of the assistant in the conversation.
        """
        self.size = size
        self.init_chat_message = None
        self.user_role = user_role
        self.assistant_role = assistant_role
        # maxlen is necessary pair, since a each new step we add an prompt and assitant answer
        self.buffer = []

    def append(self, item: Dict[str, str]) -> None:
        """
        Add a new conversation item to the chat cache.
        Arguments:
            - item (Dict[str, str]): The conversation item to add. Such as {"role": "user", "content": "Hello."}
        """
        self.buffer.append(item)
        # Remove the oldest conversation item if the cache size is exceeded.
        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)

    def init_chat(self, init_chat_message: Dict[str, str]) -> None:
        """
        Initialize the chat with an initial message (e.g. system prompt).
        """
        self.init_chat_message = init_chat_message

    def to_list(self) -> List[Dict[str, str]]:
        """
        Get the conversation buffer as a list, which is usually used as messages in LLM API.
        Returns:
            - buffer (List[Dict[str, str]]): The conversation buffer as a list.
        """
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer

    def to_pretrain_prompt(self, start: int = 0) -> str:
        """
        Convert the conversation buffer to prompt text for the pre-trained model.
        Arguments:
            - start (int): The starting index of the conversation buffer to generate the prompt.
        Returns:
            - prompt (str): The prompt text for the pre-trained LLM.
        """
        # Chinese punctuation pattern
        pattern = r'[\u3002\uff1f\uff01\uff0c\uff0e\u3001\uff1a\uff1b\uff08\uff09\u300a\u300b\u3008\u3009\u300c\u300d\u300e\u300f\u2018\u2019\u201c\u201d\u2026\u2014]$'
        prompt = ""
        for conv in self.buffer[start:]:
            content = conv.get("content", "")
            if not bool(re.search(pattern, content)):
                content += "。"
            prompt += content
        return prompt

    def to_QA_prompt(self, start: int = 0) -> str:
        """
        Convert the conversation buffer into prompt text for the question-answering (instruct) model.
        Arguments:
            - start (int): The starting index of the conversation buffer to generate the prompt.
        Returns:
            - prompt (str): The prompt text for the question-answering/instruct-tuning model.
        """
        question_format = "### 指令: {}"
        answer_format = "### 回答: {}"
        prompt = ""
        for conv in self.buffer[start:]:
            if conv.get("role", "") == self.user_role:
                prompt += question_format.format(conv.get("content", ""))
            elif conv.get("role", "") == self.assistant_role:
                prompt += answer_format.format(conv.get("content", ""))
            else:
                pass

        prompt += "### 回答: "
        return prompt

    def clear(self) -> None:
        """
        Clear the chat buffer.
        """
        self.buffer = []

    def __repr__(self) -> str:
        """
        Provides a string representation of the class.
        Returns:
            - buffer (str): The conversation buffer as a string.
        """
        return f'{self.buffer}'


class LanguageModelHandler(BaseHandler):
    """
    Handles the language model part with locally deployed models. Such as Hugging Face models.
    """
    # TODO: more intelligent transition sentence
    transition_sentence_list = [
        "好，稍等一下。",
        "嗯，明白了，等等哈。",
        "这样呀，我想想。",
    ]

    def __init__(
            self,
            stop_event: Event,
            cur_conn_end_event: Event,
            queue_in: Queue,
            queue_out: Queue,
            interruption_event: Event,
            model_name: str,
            device: str = "cuda",
            dtype: str = "float16",
            max_new_tokens: int = 128 + 32,
            temperature: float = 0.1,
            do_sample: bool = True,
            chat_size: int = 1,
            init_chat_role: Optional[str] = 'system',
            init_chat_prompt: str = "你是一个风趣幽默且聪明的智能体。",
            model_url: Optional[str] = None,  # only use for LM API
    ) -> None:
        """
        Arguments:
            - stop_event (Event): Event used to stop the processor.
            - cur_conn_end_event (Event): Event used to indicate whether the current connection has ended.
            - queue_in (Queue): Input queue.
            - queue_out (Queue): Output queue.
            - interruption_event (Event): Event used to trigger user interruption.
            - model_name (str): The name of the model to use. Such as '/home/user/Qwen2.5-7B'.
            - device (str): The device to use for processing.
            - dtype (str): The data type to use for processing. Usually 'bfloat16' or 'float16'.
            - max_new_tokens (int): The maximum number of new tokens to generate.
            - temperature (float): The temperature value for generation. 0.0 means greedy generation with no randomness.
            - do_sample (bool): Whether to use sampling for generation.
            - chat_size (int): The size of the chat cache (history).
            - init_chat_role (Optional[str]): The role of the initial chat message. Such as 'system'.
            - init_chat_prompt (str): The initial chat message.
            - model_url (Optional[str]): The URL of the model to use. Only used for the LM API \
                (e.g. https://api.openai.com).
        """
        super().__init__(stop_event, cur_conn_end_event, queue_in, queue_out)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.interruption_event = interruption_event

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(device)
        # use pipeline to control the end of generation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        # use streamer to get the generated text in real-time
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
        }
        self.user_role = "user"
        self.assistant_role = "assistant"

        self.chat = Chat(chat_size, user_role=self.user_role, assistant_role=self.assistant_role)

        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError("An initial prompt needs to be specified when setting init_chat_role.")
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        # use working event to avoid concurrent inference, there is only one inference at a time
        self.working_event = Event()

        self.warmup()

    def warmup(self) -> None:
        """
        Warm up the language model to improve responsiveness and performance in real-world use.
        """
        logger.info(f"Warming up {self.__class__.__name__}")

        # dummy_input_text = "Write me a poem about Machine Learning."
        dummy_input_text = "和另一半分手了怎么办？"
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        warmup_gen_kwargs = {
            "min_new_tokens": self.gen_kwargs["max_new_tokens"],
            "max_new_tokens": self.gen_kwargs["max_new_tokens"],
            **self.gen_kwargs
        }

        n_steps = 2

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            thread = Thread(target=self.pipe, args=(dummy_chat, ), kwargs=warmup_gen_kwargs)
            thread.start()
            new_text = "".join(list(self.streamer)) + "\n"
            logger.info(f"warm up generation: {new_text}")
        end_event.record()
        torch.cuda.synchronize()

        self.chat.clear()  # clear warm up buffer

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, inputs: Dict[str, Union[str, int, bool]]) -> Dict[str, Union[str, int, bool]]:
        """
        Process the input acquired from queue_in (from ASR/STT) and generate the output of the language model with
        the stream paradigm, i.e., yield the generated subtext in real-time.
        Arguments:
            - inputs (Dict[str, Union[str, int, bool]): The input data acquired from queue_in. The data contains the \
                str format audio transcripted data, user id(uid), bool flag to indicate whether the audio or the text \
                input from user and integer user input count.
        Returns (Yield):
            - output (Dict[str, Union[str, int, bool]]): The output data containing the transcripted question text, \
                the generated answer text, end flag for the current LLM generation, uid and the user input count.
        """
        # avoid concurrent inference, loop when working_event is set
        while self.working_event.is_set():
            time.sleep(0.1)

        self.working_event.set()
        logger.info("inference language model...")
        prompt, user_input_count, uid, audio_input = inputs["data"], inputs["user_input_count"], inputs["uid"], inputs[
            "audio_input"]
        count = 0
        # If user interruption is triggered, generate a transition sentence and yield it
        if self.interruption_event.is_set():
            self.interruption_event.clear()
            time.sleep(0.5)
            new_sentence = random.choice(self.transition_sentence_list)
            count += 1
            yield (
                {
                    'question_text': prompt if audio_input else None,
                    'answer_text': new_sentence,
                    'end_flag': False,
                    'user_input_count': user_input_count,
                    "uid": uid
                }
            )

        raw_prompt = prompt

        messages = self._before_process(prompt, user_input_count)

        logger.info(f'total input messages: {messages}')

        thread = Thread(target=self.pipe, args=(messages, ), kwargs=self.gen_kwargs)
        thread.start()

        generated_text, printable_text = "", ""
        count = 0
        for new_text in self.streamer:
            generated_text += new_text
            printable_text += new_text

            # TODO: more smart zh sentence split
            sentences = re.split(r'[。？；]', printable_text)
            sentences = [sentence for sentence in sentences if sentence.strip()]
            if not self.interruption_event.is_set() and len(sentences) > 1:
                new_sentence = sentences[0].strip() + "。"
                if count == 0:
                    yield (
                        {
                            'question_text': raw_prompt if audio_input else None,
                            'answer_text': new_sentence,
                            'end_flag': False,
                            'user_input_count': user_input_count,
                            "uid": uid
                        }
                    )
                else:
                    end_flag = sentences[1].strip() == ""
                    yield (
                        {
                            'question_text': None,
                            'answer_text': new_sentence,
                            'end_flag': end_flag,
                            'user_input_count': user_input_count,
                            "uid": uid
                        }
                    )
                count += 1
                printable_text = sentences[1]

        # don't forget last sentence
        if not self.interruption_event.is_set() and printable_text.strip() != "":
            if count == 0:
                yield {
                    'question_text': raw_prompt if audio_input else None,
                    'answer_text': printable_text,
                    'end_flag': True,
                    'user_input_count': user_input_count,
                    "uid": uid
                }
            else:
                yield {
                    'question_text': None,
                    'answer_text': printable_text,
                    'end_flag': True,
                    'user_input_count': user_input_count,
                    "uid": uid
                }

        if not self.cur_conn_end_event.is_set():
            self.chat.append({"role": "assistant", "content": generated_text})

        self.working_event.clear()
        logger.info("inference LLM over")

    def clear_current_state(self) -> None:
        """
        Clears the current state, restart/resets the chat cache.
        """
        super().clear_current_state()
        self.chat.clear()

    def _before_process(self, prompt: str, count: int) -> List[Dict[str, str]]:
        """
        Preparation chat messages before the generation process.
        Arguments:
            - prompt (str): The input prompt in current step.
            - count (int): The user input count.
        Returns:
            - messages (List[Dict[str, str]): The chat messages.
        """
        self.chat.append({"role": self.user_role, "content": prompt})
        return self.chat.to_list()


class LanguageModelAPIHandler(BaseHandler):
    """
    Handler class for interacting with the language model through the API with the OpenAI interface.
    """
    transition_sentence_list = [
        "好，稍等一下。",
        "嗯，明白了，等等哈。",
        "这样呀，我想想。",
    ]

    def __init__(
            self,
            stop_event: Event,
            cur_conn_end_event: Event,
            queue_in: Queue,
            queue_out: Queue,
            interruption_event: Event,
            model_name: str,
            max_new_tokens: int = 128 + 32,
            temperature: float = 0.1,
            do_sample: bool = True,
            chat_size: int = 1,
            init_chat_role: Optional[str] = 'system',
            init_chat_prompt: str = "你是一个风趣幽默且聪明的智能体。",
            model_url: Optional[str] = None,  # only use for LM API
            generate_questions: bool = True,  # control flag for generating questions after the reply
            **kwargs,  # for compatibility with other LMs
    ) -> None:
        """
        Arguments:
            - stop_event (Event): Event used to stop the processor.
            - cur_conn_end_event (Event): Event used to indicate whether the current connection has ended.
            - queue_in (Queue): Input queue.
            - queue_out (Queue): Output queue.
            - interruption_event (Event): Event used to trigger user interruption.
            - model_name (str): The name of the model to use. Such as 'gpt-3.5-turbo'.
            - max_new_tokens (int): The maximum number of new tokens to generate.
            - temperature (float): The temperature value for generation. 0.0 means greedy generation with no randomness.
            - do_sample (bool): Whether to use sampling for generation.
            - chat_size (int): The size of the chat cache (history).
            - init_chat_role (Optional[str]): The role of the initial chat message. Such as 'system'.
            - init_chat_prompt (str): The initial chat message.
            - model_url (Optional[str]): The URL of the model to use. Only used for the LM API \
                (e.g. https://api.openai.com).
            - generate_questions(Optional[bool]): control flag for generating questions after the reply
            - kwargs: Additional keyword arguments for compatibility with other language models, it is not used.
        """
        super().__init__(stop_event, cur_conn_end_event, queue_in, queue_out)
        self.model_name = model_name
        self.model_url = model_url
        self.interruption_event = interruption_event
        self.max_new_tokens = max_new_tokens
        self.generate_questions = generate_questions
        if do_sample:
            self.temperature = temperature
        else:
            self.temperature = 0

        self.client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=model_url)

        self.user_role = "user"
        self.assistant_role = "assistant"

        self.chat = Chat(chat_size, user_role=self.user_role, assistant_role=self.assistant_role)

        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError("An initial prompt needs to be specified when setting init_chat_role.")
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.working_event = Event()

    def process(self, inputs: Dict[str, Union[str, int, bool]]) -> Dict[str, Union[str, int, bool]]:
        """
        Process the input acquired from queue_in (from ASR/STT) and generate the output of the language model API with
        the stream paradigm, i.e., yield the generated subtext in real-time.
        Arguments:
            - inputs (Dict[str, Union[str, int, bool]): The input data acquired from queue_in. The data contains the \
                str format audio transcripted data, user id(uid), bool flag to indicate whether the audio or the text \
                input from user and integer user input count.
        Returns (Yield):
            - output (Dict[str, Union[str, int, bool]]): The output data containing the transcripted question text, \
                the generated answer text, end flag for the current LLM API generation, uid and the user input count.
        """
        # avoid concurrent inference, loop when working_event is set
        while self.working_event.is_set():
            time.sleep(0.1)

        self.working_event.set()
        logger.info("inference language model...")
        prompt, user_input_count, uid, audio_input = inputs["data"], inputs["user_input_count"], inputs["uid"], inputs[
            "audio_input"]
        count = 0
        # If user interruption is triggered, generate a transition sentence and yield it
        if self.interruption_event.is_set():
            self.interruption_event.clear()
            time.sleep(0.5)
            new_sentence = random.choice(self.transition_sentence_list)
            count += 1
            yield (
                {
                    'question_text': prompt if audio_input else None,
                    'answer_text': new_sentence,
                    'end_flag': False,
                    'user_input_count': user_input_count,
                    "uid": uid
                }
            )

        raw_prompt = prompt

        messages = self._before_process(prompt, user_input_count)

        logger.info(f'total input messages: {messages}')

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )

        generated_text, printable_text = "", ""
        for chunk in response:  # stream response
            chunk_text = chunk.choices[0].delta.content
            generated_text += chunk_text
            printable_text += chunk_text

            # TODO: more smart zh sentence split
            sentences = re.split(r'[。？；]', printable_text)
            sentences = [sentence for sentence in sentences if sentence.strip()]
            if not self.interruption_event.is_set() and len(sentences) > 1:
                new_sentence = sentences[0].strip() + "。"
                others = printable_text[len(sentences[0]) + 1:]
                if count == 0:
                    yield (
                        {
                            'question_text': raw_prompt if audio_input else None,
                            'answer_text': new_sentence,
                            'end_flag': False,
                            'user_input_count': user_input_count,
                            "uid": uid
                        }
                    )
                else:
                    end_flag = others.strip() == ""
                    yield (
                        {
                            'question_text': None,
                            'answer_text': new_sentence,
                            'end_flag': end_flag,
                            'user_input_count': user_input_count,
                            "uid": uid
                        }
                    )
                count += 1
                printable_text = others

        # don't forget last sentence
        if not self.interruption_event.is_set() and printable_text.strip() != "":
            if count == 0:
                yield {
                    'question_text': raw_prompt if audio_input else None,
                    'answer_text': printable_text,
                    'end_flag': True,
                    'user_input_count': user_input_count,
                    "uid": uid
                }
            else:
                yield {
                    'question_text': None,
                    'answer_text': printable_text,
                    'end_flag': True,
                    'user_input_count': user_input_count,
                    "uid": uid
                }

        if not self.cur_conn_end_event.is_set():
            self.chat.append({"role": "assistant", "content": generated_text})
            if self.generate_questions:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.chat.to_list() +
                    [{
                        "role": self.user_role,
                        "content": CLEANS2S_SMART_POST_QUESTION_PROMPT
                    }],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stream=False
                )
                msg = response.choices[0].message.content
                matched = re.search(CLEANS2S_SMART_POST_QUESTION_PATTERN, msg)
                if matched:
                    question1 = matched.group(1)
                    question2 = matched.group(2)
                    question3 = matched.group(3)
                    result = {"q1": question1, "q2": question2, "q3": question3}
                    yield {
                        'question_text': None,
                        'answer_text': result,
                        'end_flag': True,
                        'user_input_count': user_input_count,
                        "uid": uid
                    }

        self.working_event.clear()
        logger.info("inference LLM over")

    def clear_current_state(self) -> None:
        """
        Clears the current state, restart/resets the chat cache.
        """
        super().clear_current_state()
        self.chat.clear()

    def _before_process(self, prompt: str, count: int) -> List[Dict[str, str]]:
        """
        Preparation chat messages before the generation process.
        Arguments:
            - prompt (str): The input prompt in current step.
            - count (int): The user input count.
        Returns:
            - messages (List[Dict[str, str]): The chat messages.
        """
        self.chat.append({"role": self.user_role, "content": prompt})
        return self.chat.to_list()


class CosyVoiceTTSHandler(BaseHandler):
    """
    Handlers for text-to-speech (TTS) conversion, specifically for handling TTS tasks for the CosyVoice model.
    (arXiv: https://arxiv.org/abs/2407.05407)
    """

    def __init__(
            self,
            stop_event: Event,
            cur_conn_end_event: Event,
            queue_in: Queue,
            queue_out: Queue,
            should_listen: Event,
            interruption_event: Event,
            ref_dir: str,
            model_name: str = 'CosyVoice-300M',
            device: str = "cuda",
            dtype: str = "float32",
            compile_mode: Optional[str] = None,
            save_data: Optional[bool] = False,
    ) -> None:
        """
        Arguments:
            - stop_event (Event): Event used to stop the processor.
            - cur_conn_end_event (Event): Event used to indicate whether the current connection has ended.
            - queue_in (Queue): Input queue.
            - queue_out (Queue): Output queue.
            - should_listen (Event): Event used to indicate whether the system should listen to the user.
            - interruption_event (Event): Event used to trigger user interruption.
            - ref_dir (str): The path to the reference directory containing the reference audio files (*.wav).
            - model_name (str): The name of the model to use. Such as 'CosyVoice-300M'.
            - device (str): The device to use for TTS model.
            - dtype (str): The data type to use for processing. Usually 'bfloat16' or 'float16' or 'float32'.
            - compile_mode (Optional[str]): The compilation mode to use for the model. If None, no compilation is used.
            - save_data (Optional[bool]): Whether to save processed data into `s2d_data` directory.
        """
        super().__init__(stop_event, cur_conn_end_event, queue_in, queue_out)
        self.should_listen = should_listen
        self.interruption_event = interruption_event
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.compile_mode = compile_mode
        self.save_data = save_data
        self.working_event = Event()

        self.model = CosyVoice(model_name)
        self.stop_signal = None

        # if self.compile_mode:
        #     self.model.generation_config.cache_implementation = "static"
        #     self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)

        self.input_folder = ref_dir
        json_path = os.path.join(self.input_folder, 'ref.json')
        with open(json_path, 'r') as f:
            self.ref_list = json.load(f)
        self.warmup()

    def warmup(self) -> None:
        """
        Warm up the TTS model to improve the response speed and performance in actual use.
        """
        logger.info(f"Warming up {self.__class__.__name__}")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        n_steps = 1

        torch.cuda.synchronize()
        start_event.record()
        for item in self.ref_list:
            ref_wav_path = os.path.join(self.input_folder, 'ref_wav', item['ref_wav_path'])
            prompt_speech_16k = load_wav(ref_wav_path, 16000)

            tts_gen_kwargs = dict(
                tts_text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
                prompt_text=item['prompt_text'],
                prompt_speech_16k=prompt_speech_16k,
                stream=False,
            )
            for i in range(n_steps):
                _ = list(self.model.inference_zero_shot(**tts_gen_kwargs))
        self.ref = random.choice(self.ref_list)

        end_event.record()
        torch.cuda.synchronize()
        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self,
                inputs: Dict[str, Union[str, int, bool, dict]],
                return_np: bool = True) -> Dict[str, Union[str, np.ndarray, int, bool]]:
        """
        Process the input acquired from queue_in (from LLM) and generate the audio output of the TTS model with
        the stream paradigm, i.e., a long text sentence will be divided into several short sub-sentences to yield the
        generated sub-audio in real-time.
        Arguments:
            - inputs (Dict[str, Union[str, int, bool, dict]]): The input data acquired from queue_in. The data \
                contains the str/dict format LLM output and user id (uid), bool end flag for LLM generation, and \
                integer user input count.
        Returns (Yield):
            - output (Dict[str, Union[str, np.ndarray, int, bool]]): The output data containing the transcripted \
                question text, the LLM generated answer text, the TTS generated answer audio, end flag for the current \
                LLM&TTS (i.e. the end of current response) generation, uid and the user input count.
            - return_np (bool): Whether to return the audio data in NumPy format. If False, the audio data will be \
                torch.Tensor format.
        """
        while self.working_event.is_set():
            time.sleep(0.1)

        self.working_event.set()

        llm_sentence = inputs['answer_text']
        uid = inputs['uid']
        if isinstance(llm_sentence, dict):
            yield {
                'question_text': inputs['question_text'],
                'answer_text': llm_sentence,
                "answer_audio": 0,  # indicator for this special type return value in frontend
                "end_flag": inputs["end_flag"],
                "user_input_count": inputs['user_input_count'],
                "uid": uid
            }
        else:
            console.print(f"[green]{time.ctime()}\tASSISTANT: {llm_sentence}")
            audio_queue = Queue()
            ref_wav_path = os.path.join(self.input_folder, 'ref_wav', self.ref['ref_wav_path'])
            total_cnt = -1

            prompt_speech_16k = load_wav(ref_wav_path, 16000)
            tts_gen_kwargs = dict(
                tts_text=llm_sentence,
                prompt_text=self.ref['prompt_text'],
                prompt_speech_16k=prompt_speech_16k,
                stream=False,
            )

            def infer_fn():
                nonlocal total_cnt
                try:
                    for item in self.model.inference_zero_shot(**tts_gen_kwargs):
                        audio = item["tts_speech"]
                        # original audio returned from the model is 22.05kHz, resample it to 16kHz, which is the default
                        # sample rate of the whole pipeline
                        audio = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)(audio)
                        total_cnt = 1
                        if return_np:
                            # Transform the audio tensor to NumPy format with int16 type and range
                            audio = (audio.cpu().numpy() * 32768).astype(np.int16)
                        audio_queue.put(audio)
                        # If the interruption event is triggered, stop the TTS generation
                        if self.interruption_event.is_set() or self.cur_conn_end_event.is_set():
                            break
                    audio_queue.put(self.stop_signal)
                except Exception as e:
                    logger.error(f"TTS error {repr(e)}")
                    import traceback
                    traceback.print_exc()
                    audio_queue.put(self.stop_signal)

            thread = Thread(target=infer_fn)
            thread.start()

            i = 0
            chunks = []
            while True:
                try:
                    audio_chunk = audio_queue.get(timeout=0.2)
                    chunks.append(audio_chunk)
                except Empty:
                    if self.interruption_event.is_set() or self.cur_conn_end_event.is_set():
                        logger.info("Stop TTS generation due to the current connection is end")
                        break
                    else:
                        continue
                if self.interruption_event.is_set() or self.cur_conn_end_event.is_set():
                    logger.info("Stop TTS generation due to the current connection is end")
                    break
                if audio_chunk is self.stop_signal:
                    break
                end_flag = inputs['end_flag'] and i == total_cnt - 1
                if self.save_data:
                    readable_time = get_readable_time(time.time())
                    save_fn(
                        f"s2s_data/{uid}_{inputs['user_input_count']}_output_{readable_time}.json", {
                            "audio": np.concatenate(chunks, axis=0),
                            "text": inputs['answer_text'],
                            "time": time.ctime()
                        }
                    )
                if i == 0:
                    if pipeline_start is not None:
                        logger.info(
                            f"[green]{time.ctime()}\tTime to first user audio input: {perf_counter() - pipeline_start:.3f}"
                        )
                        console.print(
                            f"[green]{time.ctime()}\tTime to first user audio input: {perf_counter() - pipeline_start:.3f}"
                        )
                    yield {
                        'question_text': inputs['question_text'],
                        'answer_text': inputs['answer_text'],
                        "answer_audio": audio_chunk,
                        "end_flag": end_flag,
                        "user_input_count": inputs['user_input_count'],
                        "uid": uid
                    }
                else:
                    yield {
                        'question_text': None,
                        'answer_text': None,
                        "answer_audio": audio_chunk,
                        "end_flag": end_flag,
                        "user_input_count": inputs['user_input_count'],
                        "uid": uid
                    }
                i += 1

        self.should_listen.set()
        self.working_event.clear()
        logger.info("Enable should listen")

    def clear_current_state(self) -> None:
        """
        Clears the current state, restart/resets the reference audio and prompt (random choice from the reference list).
        """
        super().clear_current_state()
        self.ref = random.choice(self.ref_list)



class TTSHandler(BaseHandler):
    """
    Handlers for text-to-speech (TTS) conversion, based on https://docs.siliconflow.cn/cn/api-reference/audio
    """

    def __init__(
            self,
            stop_event: Event,
            cur_conn_end_event: Event,
            queue_in: Queue,
            queue_out: Queue,
            interruption_event: Event,
            ref_dir: str,
            model_name: str = "FunAudioLLM/CosyVoice2-0.5B",
            model_url: str = "https://api.siliconflow.cn/v1",  
            device: str = "cuda",
            dtype: str = "float32",
    ) -> None:
        """
        Arguments:
            - stop_event (Event): Event used to stop the processor.
            - cur_conn_end_event (Event): Event used to indicate whether the current connection has ended.
            - queue_in (Queue): Input queue.
            - queue_out (Queue): Output queue.
            - interruption_event (Event): Event used to trigger user interruption.
            - ref_dir (str): The path to the reference directory containing the reference audio files (*.wav).
            - model_name (str): The name of the model to use. Such as 'CosyVoice-300M'.
            - model_url(str): The base url of siliconflow api
            - device (str): The device to use for TTS model.
            - dtype (str): The data type to use for processing. Usually 'bfloat16' or 'float16' or 'float32'.
        """
        super().__init__(stop_event, cur_conn_end_event, queue_in, queue_out)
        self.interruption_event = interruption_event
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.working_event = Event()

        self.model_name = model_name
        self.model_url = model_url

        self.input_folder = ref_dir

        self.api_key = os.getenv("API_KEY")
        if self.api_key is None:
            raise ValueError("Environment variable 'API_KEY' is not set")

        self.ref_audio_cnt = 0

        self.upload_url = os.path.join(self.model_url,"/uploads/audio/voice")
        self.list_ref_url = os.path.join(self.model_url + "/audio/voice/list")
        self.tts_url = os.path.join(self.model_url + "/audio/speech")
        self.delete_ref_url = os.path.join(self.model_url, "/audio/voice/deletions")

        wav_files = glob.glob(os.path.join(self.input_folder, "*.wav"))

        for wav_file in wav_files:
            txt_file = os.path.splitext(wav_file)[0] + ".txt"

            if os.path.exists(txt_file):
                with open(txt_file, "r", encoding="utf-8") as f:
                    text_content = f.read().strip()
            else:
                text_content = ""

            self.upload_reference_audio((wav_file, text_content))

        ref_audio_list = self.list_reference()
        logger.info("All reference audio uri are listed below:")
        for ref_audio in ref_audio_list:
            logger.info(ref_audio['uri'])

    def upload_reference_audio(self, reference_audio:Tuple[str,str]) -> Dict[str,str]:
        """upload reference audio to siliconflow api

        Args:
            reference_audio (Tuple[str,str]): The first element is the absolute path of the reference audio file, and the second element is the corresponding text of the reference audio.

        Returns:
            Dict[str,str]: the uri of the reference audio file like {'uri': 'speech:hus-test-2435767:ng9rdjx3ph:yuensltrtyfzysrktlsp'}
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        audio_file = {"file":open(reference_audio[0],"rb")}
        data = {
            "model": self.model_name,
            "customName": f"ref_audio_{reference_audio[0]}_{self.ref_audio_cnt}",
            "text": reference_audio[1]
        }
        self.ref_audio_cnt += 1
        response = requests.post(self.upload_url, headers=headers, files=audio_file, data=data)
        return response.json() 
    
    def list_reference(self) -> List[Dict[str,str]]:
        """list all the uploaded reference audio file

        Returns:
        List[Dict[str,str]]: all the uploaded reference audio with its model, customName, text and uri
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.request("GET", self.list_ref_url, headers=headers)
        response = response.json()
        return response['result']

    def process(self, text, ref_voice, save_path, stream = False, 
                config: Dict[str, Union[str, int, bool, dict]] = dict()) -> None:
        """transform text to speech, now text is str not list

        Args:
            text (str): User input text to be transformed
            ref_voice (str): The uri of the reference voice to be applied on the text
            save_path (str): The save path of the output audio path
            stream (bool, optional): whether the output is in stream format. Defaults to False. Not supported now.
            config (Dict[str, Union[str, int, bool, dict]], optional): configuration of the TTS model including response_format(default "mp3"), sample_rate(default 32000), speed(default 1) and gain(default 0). Defaults to None.
        """
        while self.working_event.is_set():
            time.sleep(0.1)
        self.working_event.set()

        # process output_file_name
        if os.path.isfile(save_path) or '.' in os.path.basename(save_path):
            file_path = save_path
        else:
            os.makedirs(save_path, exist_ok=True)
    
            file_name = f"{text[:5] if len(text) >= 5 else text}.{config.get('response_format', 'mp3')}"
            file_name = "".join(c for c in file_name if c.isalnum() or c in (' ', '.', '_')).rstrip()
            
            file_path = os.path.join(save_path, file_name)

        def text2audio() :
            payload= {
                "model": self.model_name,
                "input": text,
                "voice": ref_voice, # FunAudioLLM/CosyVoice2-0.5B:benjamin
                "response_format": config.get("response_format", "mp3"),
                "sample_rate": config.get("sample_rate", 32000),
                "stream": stream,
                "speed": config.get("speed", 1),
                "gain": config.get("gain", 0)
            }

            headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
            }
            
            response = requests.request("POST", self.tts_url, json=payload, headers=headers)

            if response.status_code == 200:
                with open(file_path,'wb')as file:
                    file.write(response.content)
                logger.info(f"file saved at {file_path}")

                try:
                    audio_bytes = BytesIO(response.content)
                    waveform, sr = torchaudio.load(audio_bytes, format=config.get("response_format", "mp3"))

                    # assert sr == sample_rate, "sample rate does not match"
                    self._last_audio_array = waveform.numpy()

                except Exception as e:
                    logger.error(f"Error decoding audio with torchaudio: {e}")
                    self._last_audio_array = None
            else:
                logger.info(f"error: status code {response.status_code}")
                self._last_audio_array = None

        if self.interruption_event.is_set() or self.cur_conn_end_event.is_set():
            logger.info("Stop TTS generation due to the current connection is end")
            self.working_event.clear()
            return None

        thread = Thread(target=text2audio)
        thread.start()
        thread.join()

        if self.interruption_event.is_set() or self.cur_conn_end_event.is_set():
            logger.info("Stop TTS generation due to the current connection is end")
            self.working_event.clear()
            return None

        self.working_event.clear()

        return self._last_audio_array

    def clear_current_state(self) -> None:
        """
        Clears the current state, restart/resets the reference audio and prompt (random choice from the reference list).
        """
        super().clear_current_state()
        self.ref_audio_cnt = 0

        def delete_ref_audio():
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json"}
            ref_audio_list = self.list_reference()
            for ref_audio in ref_audio_list:
                payload = {"uri": ref_audio['uri']}
                response = requests.request("POST", self.delete_ref_url, json=payload, headers=headers)
        
        delete_ref_audio()



class ASRHandler(BaseHandler):
    """
    Handlers for text-to-speech (TTS) conversion, based on https://docs.siliconflow.cn/cn/api-reference/audio
    """

    def __init__(
            self,
            stop_event: Event,
            cur_conn_end_event: Event,
            queue_in: Queue,
            queue_out: Queue,
            interruption_event: Event,
            model_name: str = "FunAudioLLM/SenseVoiceSmall",
            model_url: str = "https://api.siliconflow.cn/v1/audio/transcriptions",  
            device: str = "cuda",
            dtype: str = "float32",
    ) -> None:
        """
        Arguments:
            - stop_event (Event): Event used to stop the processor.
            - cur_conn_end_event (Event): Event used to indicate whether the current connection has ended.
            - queue_in (Queue): Input queue.
            - queue_out (Queue): Output queue.
            - interruption_event (Event): Event used to trigger user interruption.
            - model_name (str): The name of the model to use. Such as 'CosyVoice-300M'.
            - model_url(str): The base url of siliconflow api
            - device (str): The device to use for TTS model.
            - dtype (str): The data type to use for processing. Usually 'bfloat16' or 'float16' or 'float32'.
        """
        super().__init__(stop_event, cur_conn_end_event, queue_in, queue_out)
        self.interruption_event = interruption_event
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.working_event = Event()

        self.model_name = model_name
        self.model_url = model_url

        self.api_key = os.getenv("API_KEY")
        if self.api_key is None:
            raise ValueError("Environment variable 'API_KEY' is not set")


    def process(self, file_path):
        while self.working_event.is_set():
            time.sleep(0.1)
        self.working_event.set()

        def audio2text():
            with open(file_path, "rb") as audio_file:
                files = {
                    "file": ("audio_file.wav", audio_file, "audio/wav"),
                    "model": (None, self.model_name, "text/plain")
                }

                headers = {
                "Authorization": f"Bearer {self.api_key}"
                }
                
                response = requests.post(self.model_url, files=files, headers=headers)
                response = response.json()
                return response['text']

        if self.interruption_event.is_set() or self.cur_conn_end_event.is_set():
            logger.info("Stop TTS generation due to the current connection is end")

        response = audio2text()

        self.working_event.clear()

        return response

    def clear_current_state(self) -> None:
        """
        Clears the current state, restart/resets the reference audio and prompt (random choice from the reference list).
        """
        super().clear_current_state()


def main(args) -> None:
    """
    Main pipeline function for Speech-to-Speech interaction.
    """
    random.seed(time.time())
    torch.manual_seed(0)
    # torch compile logs
    # torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)
    if args.save_data:
        os.makedirs("s2s_data", exist_ok=True)

    # 1. Build the pipeline
    stop_event = Event()
    # used to stop putting received audio chunks in queue until all setences have been processed by the TTS
    should_listen = Event()
    # used to indicate whether the current connection is end
    cur_conn_end_event = Event()
    # used to control the user's interruption
    interruption_event = Event()
    send_audio_chunks_queue = Queue()
    spoken_prompt_queue = Queue()
    text_prompt_queue = Queue()
    lm_response_queue = Queue()

    stt = ParaFormerSTTHandler(
        stop_event,
        cur_conn_end_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
        model_name=args.stt_model_name,
        device=args.device,
        dtype=args.stt_dtype,
        save_data=args.save_data,
    )
    lm_cls = LanguageModelAPIHandler if args.enable_llm_api else LanguageModelHandler
    lm = lm_cls(
        stop_event,
        cur_conn_end_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        interruption_event=interruption_event,
        model_name=args.lm_model_name,
        model_url=args.lm_model_url,
        device=args.device,
        dtype=args.lm_dtype,
        max_new_tokens=args.max_new_tokens,
        chat_size=args.chat_size,
        init_chat_role=args.init_chat_role,
        init_chat_prompt=args.init_chat_prompt,
    )
    tts = CosyVoiceTTSHandler(
        stop_event,
        cur_conn_end_event,
        queue_in=lm_response_queue,
        queue_out=send_audio_chunks_queue,
        should_listen=should_listen,
        interruption_event=interruption_event,
        model_name=args.tts_model_name,
        device=args.device,
        dtype=args.tts_dtype,
        ref_dir=args.ref_dir,
        save_data=args.save_data,
    )

    recv_handler = SocketVADReceiver(
        stop_event,
        cur_conn_end_event,
        spoken_prompt_queue,
        should_listen,
        interruption_event,
        host=args.recv_host,
        port=args.recv_port,
        chunk_size=args.chunk_size,
        thresh=args.vad_thresh,
        sample_rate=args.sample_rate,
        min_silence_ms=args.min_silence_ms,
        min_speech_ms=args.min_speech_ms,
        enable_interruption=args.enable_interruption,
    )

    send_handler = SocketSender(
        stop_event,
        cur_conn_end_event,
        send_audio_chunks_queue,
        host=args.send_host,
        port=args.send_port,
    )

    # 2. Run the pipeline
    pipeline_manager = ThreadManager([tts, lm, stt, recv_handler, send_handler])
    try:
        pipeline_manager.start()
        pipeline_manager.join()
    except KeyboardInterrupt:
        pipeline_manager.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CleanS2S Arguments")
    # Common
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Whether to save audio/text data during processing. Data will be saved into `s2s_data` directory."
    )
    # Socket Recevier and VAD
    parser.add_argument(
        "--recv_host",
        type=str,
        default="localhost",
        help=
        "The host IP address for the socket connection. Default is '0.0.0.0' which binds to all available interfaces on the host machine."
    )
    parser.add_argument(
        "--recv_port",
        type=int,
        default=9001,
        help="The port number on which the socket server listens. Default is 12346."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2048,
        help="The size of each data chunk to be sent or received over the socket. Default is 1024 bytes."
    )
    parser.add_argument(
        "--vad_thresh",
        type=float,
        default=0.3,
        help=
        "The threshold value for voice activity detection (VAD). Values typically range from 0 to 1, with higher values requiring higher confidence in speech detection."
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="The sample rate of the audio in Hertz. Default is 16000 Hz, which is a common setting for voice audio."
    )
    parser.add_argument(
        "--min_silence_ms",
        type=int,
        default=1200,
        help=
        "Minimum length of silence intervals to be used for segmenting speech. Measured in milliseconds. Default is 1000 ms."
    )
    parser.add_argument(
        "--min_speech_ms",
        type=int,
        default=400,
        help=
        "Minimum length of speech segments to be considered valid speech. Measured in milliseconds. Default is 500 ms."
    )
    parser.add_argument(
        "--enable_interruption",
        action="store_true",
        help="Whether to support the user's speech interruption. Default is False."
    )
    # Socket Sender
    parser.add_argument(
        "--send_host",
        type=str,
        default="localhost",
        help=
        "The host IP address for the socket connection. Default is '0.0.0.0' which binds to all available interfaces on the host machine."
    )
    parser.add_argument(
        "--send_port",
        type=int,
        default=9002,
        help="The port number on which the socket server listens. Default is 12346."
    )
    # General Model
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device type on which the model will run. Default is 'cuda' for GPU acceleration."
    )
    # STT Model
    parser.add_argument(
        "--stt_dtype",
        type=str,
        default="float16",
        help=
        "The PyTorch data type for the STT model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
    )
    parser.add_argument("--stt_model_name", type=str, help="The pretrained STT model to use.")
    # Language Model
    parser.add_argument(
        "--enable_llm_api",
        action="store_true",
        help="Whether to use language model API, otherwise, it will use the local-deployed model."
    )
    parser.add_argument(
        "--lm_dtype",
        type=str,
        default="bfloat16",
        help=
        "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
    )
    parser.add_argument(
        "--lm_model_name", type=str, help="The pretrained language model to use. Such as `deepseek-chat`"
    )
    parser.add_argument(
        "--lm_model_url", type=str, help="The pretrained language model to use. Such as `https://api.deepseek.com`"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128 + 32, help="Max output new token numbers of language model."
    )
    parser.add_argument(
        "--init_chat_role",
        type=str,
        default='system',
        help="Initial role for setting up the chat context. Default is 'system'."
    )
    parser.add_argument(
        "--init_chat_prompt",
        type=str,
        default=CLEANS2S_SMART_INTERACTION_PROMPT,
        help="The initial chat prompt to establish context for the language model.'"
    )
    parser.add_argument(
        "--chat_size",
        type=int,
        default=1,
        help="Number of interactions assistant-user to keep for the chat. None for no limitations."
    )
    # TTS Model
    parser.add_argument(
        "--tts_model_name",
        type=str,
        help="The pretrained TTS model to use. Such as the local path `/home/user/CosyVoice-300M`"
    )
    parser.add_argument(
        "--tts_dtype",
        type=str,
        default="float32",
        help=
        "The PyTorch data type for the TTS model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
    )
    parser.add_argument(
        "--ref_dir", type=str, help="The folder directory path of TTS reference audio and related prompt."
    )
    args = parser.parse_args()

    main(args)
