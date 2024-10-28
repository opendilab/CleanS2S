from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
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

import numpy as np
import torch
import nltk
import asyncio
import websockets
from rich.console import Console
# ASR
from funasr import AutoModel
# TTS
import torchaudio
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
# CURRENT_DIR = Path(__file__).resolve().parent
# os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")
# torch._inductor.config.fx_graph_cache = True
# # mind about this parameter ! should be >= 2 * number of padded prompt sizes for TTS
# torch._dynamo.config.cache_size_limit = 15

global logger
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.info(f'BEGIN LOGGER {__name__}')
console = Console()
global pipeline_start
pipeline_start = None


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
            max_speech_ms: float = 5000,
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
            max_speech_ms=max_speech_ms,
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
                                self.queue_out.put({"data": vad_result, "user_input_count": self.user_input_count, "uid": uid})
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
            if duration_ms < self.min_speech_ms:
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
            compile_mode: Optional[str] = None,
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
            - compile_mode (Optional[str]): The compile mode to use for processing the model.
        """
        super().__init__(stop_event, cur_conn_end_event, queue_in, queue_out)
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.compile_mode = compile_mode

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
            sentence_timestamp=False,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
            # spk_mode="punc_segment"
        )

        # compile
        # if self.compile_mode:
        #     self.model.generation_config.cache_implementation = "static"
        #     self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)
        # self.warmup()

    def warmup(self) -> None:
        """
        Model warm-up to improve the model's responsiveness and performance in real-world use.
        """
        raise NotImplementedError

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
        logger.info("inference ASR pacaformer...")
        spoken_prompt, user_input_count, uid = inputs["data"], inputs["user_input_count"], inputs["uid"]

        global pipeline_start
        pipeline_start = perf_counter()
        
        # user directly send text question
        if isinstance(spoken_prompt, str):
            console.print(f"[yellow]{time.ctime()}\tUSER: {spoken_prompt}")
            yield {"data": spoken_prompt, "user_input_count": user_input_count, "uid": uid, "audio_input": False}
        else:
            res = self.model.generate(input=spoken_prompt, batch_size_s=300, batch_size_threshold_s=60)
            try:
                # pred_text = "".join([t["text"] for t in res[0]["sentence_info"]])
                pred_text = res[0]["text"]
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"ASR get nothing: {repr(e)}")
                return None

            logger.info("finish ASR paraformer inference")
            console.print(f"[yellow]{time.ctime()}\tUSER: {pred_text}")

            yield {"data": pred_text, "user_input_count": user_input_count, "uid": uid, "audio_input": True}


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
        """
        super().__init__(stop_event, cur_conn_end_event, queue_in, queue_out)
        self.should_listen = should_listen
        self.interruption_event = interruption_event
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.compile_mode = compile_mode
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
                inputs: Dict[str, Union[str, int]],
                return_np: bool = True) -> Dict[str, Union[str, np.ndarray, int, bool]]:
        """
        Process the input acquired from queue_in (from STT) and generate the audio output of the TTS model with
        the stream paradigm, i.e., a long text sentence will be divided into several short sub-sentences to yield the
        generated sub-audio in real-time.
        Arguments:
            - inputs (Dict[str, Union[str, int]]): The input data acquired from queue_in. The data contains the \
                str format STT output and user id (uid), and integer user input count.
        Returns (Yield):
            - output (Dict[str, Union[str, np.ndarray, int, bool]]): The output data containing the transcripted \
                question data, the TTS generated answer audio, end flag for the current \
                TTS (i.e. the end of current response) generation, uid and the user input count.
            - return_np (bool): Whether to return the audio data in NumPy format. If False, the audio data will be \
                torch.Tensor format.
        """
        while self.working_event.is_set():
            time.sleep(0.1)

        self.working_event.set()

        llm_sentence = inputs['data']
        uid = inputs['uid']
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
                    # If the current conncection end event is triggered, stop the TTS generation
                    if self.cur_conn_end_event.is_set():
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
        while True:
            try:
                audio_chunk = audio_queue.get(timeout=0.2)
            except Empty:
                if self.cur_conn_end_event.is_set():
                    logger.info("Stop TTS generation due to the current connection is end")
                    break
                else:
                    continue
            if self.cur_conn_end_event.is_set():
                logger.info("Stop TTS generation due to the current connection is end")
                break
            if audio_chunk is self.stop_signal:
                break
            end_flag = True
            if i == 0:
                if pipeline_start is not None:
                    logger.info(f"[green]{time.ctime()}\tTime to first user audio input: {perf_counter() - pipeline_start:.3f}")
                    console.print(f"[green]{time.ctime()}\tTime to first user audio input: {perf_counter() - pipeline_start:.3f}")
                yield {
                    'question_text': None,
                    'answer_text': inputs['data'],
                    "answer_audio": audio_chunk,
                    "end_flag": end_flag,
                    "user_input_count": inputs['user_input_count'],
                    "uid": uid
                }
            else:
                yield {
                    'question_text': None,
                    'answer_text': inputs['data'],
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


def main(args) -> None:
    """
    Main pipeline function for Speech-to-Speech interaction.
    """
    random.seed(time.time())
    torch.manual_seed(0)
    # torch compile logs
    # torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

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

    stt = ParaFormerSTTHandler(
        stop_event,
        cur_conn_end_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
        model_name=args.stt_model_name,
        device=args.device,
        dtype=args.stt_dtype,
    )
    tts = CosyVoiceTTSHandler(
        stop_event,
        cur_conn_end_event,
        queue_in=text_prompt_queue,
        queue_out=send_audio_chunks_queue,
        should_listen=should_listen,
        interruption_event=interruption_event,
        model_name=args.tts_model_name,
        device=args.device,
        dtype=args.tts_dtype,
        ref_dir=args.ref_dir,
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
    pipeline_manager = ThreadManager([tts, stt, recv_handler, send_handler])
    try:
        pipeline_manager.start()
        pipeline_manager.join()
    except KeyboardInterrupt:
        pipeline_manager.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CleanVC Arguments")
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
        default=0.25,
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
        default=1000,
        help=
        "Minimum length of silence intervals to be used for segmenting speech. Measured in milliseconds. Default is 1000 ms."
    )
    parser.add_argument(
        "--min_speech_ms",
        type=int,
        default=200,
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
