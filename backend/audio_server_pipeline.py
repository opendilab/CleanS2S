from typing import Dict, List, Optional, Union, Any
from threading import Event, Thread
from queue import Queue, Empty
import torch
import glob
import os
import time
import logging
import requests
from typing import Tuple,Dict,List

from s2s_server_pipeline import BaseHandler   


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)



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
    
            file_name = f"{text[:5] if len(text) >= 5 else text}.{config.get("response_format", "mp3")}"
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
                print(f"file saved at {file_path}")
            else:
                print(f"error: status code {response.status_code}")

        if self.interruption_event.is_set() or self.cur_conn_end_event.is_set():
            logger.info("Stop TTS generation due to the current connection is end")

        thread = Thread(target=text2audio)
        thread.start()

        if self.interruption_event.is_set() or self.cur_conn_end_event.is_set():
            logger.info("Stop TTS generation due to the current connection is end")

        self.working_event.clear()

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





    