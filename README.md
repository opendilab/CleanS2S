# CleanS2S

English | [简体中文(Simplified Chinese)](https://github.com/opendilab/CleanS2S/blob/main/README.zh.md) 

**CleanS2S** is a Speech-to-Speech (**S2S**) prototype agent that provides high-quality and streaming interactions in the single-file implementation. This design is simple and clean, aiming to provide a 
Chinese interactive prototype agent like the GPT-4o style. This project wants to let users directly experience the power of Linguistic User Interface (**LUI**) and quickly explore/vailidate the potential of the S2S pipeline for researchers.

Here are some live demos of CleanS2S:

TBD

## Outline

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

### 📜 Single-file implementation

Every detail about a kind of agent pipeline is put into a single standalone file. There is no extra burden to configure the dependencies and understand the project file structure.
So it is a great reference implementation to read for folks who want to quickly have a glance at the S2S pipeline and directly vailidate novel ideas on top of it.
All the pipeline implementations are easy to modify and extend, and the user can quickly change the model (e.g. LLM) they like, add new components, or customize the pipeline.

### 🎮 Real-time streaming interface

The whole S2S pipeline is mainly composed of `ASR` (Automatic Speech Recognition, or named Speech to Text), `LLM` (Large Language Model), and `TTS` (Text to Speech), together with two `WebSockets` components Receiver (contains VAD) and Sender.
The pipeline is designed to be real-time streaming, which means the user can interact with the agent in real-time like a human-to-human conversation. All the audio and text information is streamed sent and received through the WebSocket.
To achieve this, we utilize multi-threading and queueing mechanisms to ensure the streaming process and avoid the blocking issue. All the components are designed to be asynchronous and non-blocking, processing the data from input queue and output result into another queue.

### 🧫 Full-duplex interaction with interruptions

Based on the powerful mechanisms provided by [WebSockets](https://websockets.readthedocs.io/en/stable/), the pipeline supports full-duplex interaction, which means the user can speak and listen to the agent at the same time.
Furthermore, the pipeline supports interruptions, which means the user can interrupt the agent at any time during the conversation with a new sppech input. The agent will stop current processing and start to process the new input with the context of the previous conversations and interruptions.
Besides, we find the "assistant-style" and "turned-based" response usually used in chatbot is one of the most important drawbacks for human-like conversation. We add more interesting strategies for the agent to make the conversation more interactive and engaging. 

### 🌍 Complemented with Web Search and RAG

The pipeline is further enhanced by the integration of web search capabilities and the Retrieval-Augmented Generation (RAG) model. 
These features provide the agent with the ability to not only process and respond to user inputs in real-time but also to access and incorporate external information from the web into its responses. 
This provides room for expansion and agility in answering various practical questions raised by users.
  - The WebSearchHelper class is responsible for conducting online searches based on user queries or to gather additional information relevant to the conversation. This allows the agent to reference up-to-date or external data, enhancing the richness and accuracy of its responses.
  - The RAG class implements a retrieval-augmented generation approach, which first retrieves relevant information from a database and then uses that information to generate responses. This two-step process ensures that the agent's replies are grounded in relevant, factual data, leading to more informed and contextually appropriate interactions.

<table>
  <tr>
    <th>Case</th>
    <td>
      <strong>LanguageModelHandler</strong><br>
      <span style="color: grey; font-size: smaller;">(Qwen-7b)</span>
    </td>
    <td>
      <strong>RAGLanguageModelHelper</strong><br>
      <span style="color: grey; font-size: smaller;">(deepseek-api)</span>
    </td>
  </tr>
  <tr>
    <td>Case 1</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Case 2</td>
    <td>Row 2 Col 2</td>
    <td>Row 2 Col 3</td>
  </tr>
  <tr>
    <td>Case 3</td>
    <td>Row 3 Col 2</td>
    <td>Row 3 Col 3</td>
  </tr>
</table>
For more test case, please click [WebSockets](tbd).

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
python3 -u s2s_server_pipeline.py \
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
> ℹ️ **Support for customized LLM**: Here we use deepseek-chat as the default LLM API, you can also change to other LLM API follow the OpenAI interface. (modify the `--lm_model_name` and `--lm_model_url`, set your own API key)

> ℹ️ **Support for other customizations**: You can refer to the parameters list implemented by the `argparse` library in the backend pipeline file (e.g. `s2s_server_pipeline.py`) to customize it according to your own needs.
All the parameters are well-documented in their help attributes and easy to understand.

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
- [ ] More prompts and RAG strategies
- [ ] Customizerd example characters
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
    title={CleanS2S: High-quality and streaming Speech-to-Speech interactive agent in a single file},
    author={Niu, Yazhe and Hu, Shuai and Chen, Yun},
    publisher={GitHub},
    howpublished={\url{https://github.com/opendilab/CleanS2S}},
    year={2024},
}
```

## License

CleanS2S released under the Apache 2.0 license.
