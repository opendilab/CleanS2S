## Introduction to CleanS2S
**CleanS2S** is a Speech-to-Speech (S2S) prototype agent that provides high-quality and streaming interactions in the single-file implementation. This design is simple and clean, aiming to provide a 
Chinese interactive prototype agent like the GPT-4o style. This project wants to let users directly experience the power of Linguistic User Interface (LUI) and quickly explore/vailidate the potential of the S2S pipeline for researchers.

Here are some live demos of CleanS2S:

TBD

## Outline

- [Introduction to CleanS2S](#introduction-to-cleans2s)
- [Outline](#outline)
- [Features](#features)
- [Get Started](#get-started)
  - [Backend (Server)](#backend-server)
  - [Frontend (Client)](#frontend-client)
- [Roadmap](#roadmap)
- [Support and Get Involved](#support-and-get-involved)
- [Acknowledgements](#acknowledgements)
- [Citing CleanS2S](#citing-cleans2s)
- [License](#license)


## Features

### Single-file implementation

### Real-time streaming interface

### Full-duplex interaction with interruptions

### Complemented with Web Search and RAG


## Get started

### Backend (Server)

#### Installation
```bash
## clone the repository
git clone https://github.com/opendilab/CleanS2S.git
cd CleanS2S/backend
pip install -r requirements.txt
```

#### Downloading models
Here are 4 necessary models you need to download (3 ASR + 1 TTS), you can download them from the following links and put them in your own proper directory.
- ASR: [paraformer-zh](https://huggingface.co/funasr/paraformer-zh), [ct-punc](https://huggingface.co/funasr/ct-punc), [fsmn-vad](https://huggingface.co/funasr/fsmn-vad)
- TTS: [CosyVoice-300M](https://github.com/FunAudioLLM/CosyVoice?tab=readme-ov-file#install)

For LLM, we use LLM API by default, you can also follow the instructions [here]() to customize your own local LLM (such as DeepSeek-V2.5, Qwen2.5, etc.).

You also need to prepare a reference audio directory, which contains the reference audios for the prosody and timbre transfer. Here we prepare a [sample reference audio directory]() in this repository.
If you want to use your own reference audio, you need to keep it in the same format as the sample reference audio directory. And the audio should be 10~20 seconds long with clear pronunciation.


#### Running the server

Here is an example of running the server with the default settings:
```bash
export LLM_API_KEY=<your-deepseek-api-key>
python3 -u s2s_pipeline.py \
        --recv_host 0.0.0.0 \
        --send_host 0.0.0.0 \
        --stt_model_name <your-asr-path> \
        --enable_llm_api \
        --lm_model_name "deepseek-chat" \
        --lm_model_url "https://api.deepseek.com" \
        --tts_model_name <your-tts-path> \
        --ref_dir <ref-audio-path> \
        --enable_interruption
```
P.S. Here we use deepseek-chat as the default LLM API, you can also change to other LLM API follow the OpenAI interface. (modify the `--lm_model_name` and `--lm_model_url`, set your own API key)

### Frontend (Client)

We recommend using the `Docker image` for install and run the client. Here is the specific steps:

```bash
## run the basic docker image
docker run -it -p 3001:3001 amazonlinux:2023.2.20231011.0 sh
```

```bash
## install the necessary packages
dnf install vim git nodejs -y
npm install -g pnpm
git clone https://github.com/opendilab/CleanS2S.git
cd CleanS2S/frontend_nextjs
pnpm install
```

```bash
## run the client
pnpm dev --port 3001
```

Then you can visit the client at `http://localhost:3001` in your browser (Chrome is recommended).

P.S.: If you want to run the client locally, you should install node.js and pnpm first, then use pnmp to install the necessary packages and run the client.

## Roadmap
- [ ] Inference speed optimization
- [ ] Multi-user support for backend
- [ ] More interesting interraction and challenging mechanism
- [ ] e2e

## Support and get involved

We appreciate all the feedbacks and contributions. Feel free to ask questions. Posting in Github Issues and PRs are also welcome.

- [File an issue](https://github.com/opendilab/CleanS2S/issues/new/choose) on Github
- Discuss on CleanS2S [discord channel](https://discord.gg/dkZS2JF56X)
- Discuss on OpenDILab's WeChat group (i.e. add us on WeChat: ding314assist)


## Acknowledgements
- We thank [speech-to-speech](https://github.com/huggingface/speech-to-speech) for first open-sourcing the English speech-to-speech pipeline.
- We thank [funasr](https://github.com/modelscope/FunASR) and [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for open-sourcing high-quality Chinese ASR/TTS models.
- We thank [HumeAI](https://github.com/HumeAI) for open-sourcing a series of frontend components.

## Citing CleanS2S
```latex
@misc{CleanS2S,
    title={DI-engine: A Universal AI System/Engine for Decision Intelligence},
    author={Niu, Yazhe and Hu, Shuai and Chen, Yun},
    publisher={GitHub},
    howpublished={\url{https://github.com/opendilab/CleanS2S}},
    year={2024},
}
```

## License

CleanS2S released under the Apache 2.0 license.
