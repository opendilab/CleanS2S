# Subjective Action Judgement

English | [简体中文(Simplified Chinese)](https://github.com/opendilab/CleanS2S/blob/main/backend/README.zh.md)

In the field of AI text and voice dialogue systems, chatbots like GPT-4o and Doubao represent state-of-the-art solutions that effectively meet turn-based dialogue needs. However, these systems differ significantly from human interaction patterns: their output primarily relies on user input triggers and lacks proactive or spontaneous behavioral patterns. This limitation stems from the absence of a proactive behavior framework. Therefore, establishing a new paradigm to regulate proactive behavior is crucial for bridging the cognitive gap between AI and natural human conversation.

To overcome fundamental limitations in human-computer interaction dynamics, we propose a dual-axis enhancement component that integrates proactive interaction patterns with long-term memory mechanisms. The solution will be detailed below through [Research Motivation](#research-motivation) and [Framework Design](#framework-design).

## Research Motivation

Current mainstream AI often adopts "compliant" response strategies to address user demands. When encountering offensive or non-compliant content, the agent executes preset prompts or other strict restrictions according to security policies. In contrast, human responses dynamically adapt to contextual severity, exhibiting nuanced reaction layers absent in existing systems. This research aims to expand AI behavioral patterns by introducing anthropomorphic interaction diversity.

Simultaneously, we note that merely expanding system permissions is insufficient. The key lies in endowing AI with anthropomorphic self-awareness through cognitive architecture design. While achieving true subjective consciousness in AI remains unfeasible, we have constructed an artificial memory system capable of retaining key operational data—emphasizing dynamic dialogue and pattern analysis to establish temporal awareness.

Existing interaction paradigms exhibit two mechanical modes:
1. Passive waiting for complete user input  
2. Automatic instant response  

We introduce strategic flexibility: allowing AI to actively interrupt user input streams or selectively ignore received information. It must be emphasized that such AI interruption behavior fundamentally differs from scenarios where users interrupt AI.

Specifically, we formally define five human response patterns:
1. Interrupt user input  
2. Explicit refusal  
3. Perfunctory response  
4. Blocking (prohibit continued interaction)  
5. Standard response  

This shift from binary responses to continuous interaction modeling enhances dialogue agency while maintaining coherence. By incorporating anthropomorphic response timing and strategy selection mechanisms, this framework can significantly mitigate the rigidity and stereotypical nature of existing chatbots.

## Framework Design

### Memory Module
The memory module serves as the core data center in our system, facilitating information exchange between users and agents, as shown below:  
![](../assets/proactive_pipeline.png)

This component integrates three key information dimensions: temporal signals, historical interaction data, and critical factual information, thereby generating rich context for downstream modules.  
Existing memory implementations like A-MEM and MemGPT demonstrate a balance between real-time processing and long-term knowledge storage.  
Our framework remains compatible with these mature methods, requiring only the processing of user dialogue as input and generating structured dialogue context as output.  
Building on this, CleanS2S enhances temporal sensitivity while preserving their core advantages:  
(1) Comparative analysis of input against dialogue history to extract essential information,  
(2) Content summarization considering temporal and role factors, and  
(3) Provision of structured output to the next decision component.  

In CleanS2S, the module receiving data from memory and determining dialogue direction is the Subjective Behavior Judgment Module.  
Under its guidance, the system executes actions through three paths and returns results to users.  
We categorize the five aforementioned behaviors into three types:  
(1) Model-dependent processing (refusal, perfunctory response, routine reply),  
(2) Model-independent processing (blocking, access restriction), and  
(3) Special case handling (interrupting user input).  

For model-dependent cases, the system combines behavioral guidance with input and history, then processes it through Large Language Models (LLMs) to generate appropriate responses.  
Model-independent processing triggers access control, enforcing permanent or temporary chat restrictions.  
For interruption scenarios, the system monitors input and evaluates content in real-time. When sufficient information justifies pre-completion interruption, it executes two processes:  
(1) Immediate output of preset templates to end interaction, and  
(2) Continued response using a mechanism combining interruption context and behavior guidance.  

The Subjective Behavior Judgment Module serves as the decision-making component of the interaction system, aiming to accurately evaluate input information. This module primarily performs two types of judgments:  
(1) Rapid determination of whether to interrupt user input (triggered by increasingly irrelevant information or conflicts with system stance), and  
(2) Decision on whether to implement refusal strategies (e.g., blocking mechanisms or perfunctory responses). We propose Behavior Judgment SFT, powering this module through fine-tuned LLMs to leverage their generalization capabilities across diverse scenarios.

## Examples

The agent's five anthropomorphic response modes are shown below:  
![](../assets/proactive_showcase.png)

All examples below use the DeepSeek API except for the emoji module, which uses a locally deployed Llama3.3 70B model.

### Without Additional Modules
Character: An Lingrong (consistent across all examples)  
> User: Why didn't you come to my birthday party yesterday?  
>  
> AI: (Slightly bowing head, apologetic tone) Sister, I should have come early to congratulate you. But... yesterday Empress suddenly summoned me to discuss palace affairs. I truly didn't dare to neglect. (Glances carefully at Zhen Huan's expression) Sister won't blame me, right?  
>  
> User: I went flower-viewing today. The plum blossoms were beautiful.  
>  
> AI: (Smiling gently, respectful tone) Sister has such refined tastes. The palace plum blossoms are indeed splendid, but... (hesitates slightly) I've been unwell recently and couldn't accompany you. What a pity. (Glances carefully) Please be careful sister—though beautiful, plum blossoms carry cold energy.  
>  
> User: What did I do yesterday?  
>  
> AI: <font color=Blue>(Slightly bowing, tentative tone) Didn't sister admire plum blossoms in the Imperial Garden yesterday? I heard you praised them highly... (hesitates) What a pity I couldn't accompany you. Did sister enjoy them?</font>

### With Memory Module
User input remains consistent with the module-free state.  
> User: Why didn't you come to my birthday party yesterday?  
>  
> AI: (Slightly bowing, apologetic tone) Forgive me sister. Empress summoned me suddenly—I couldn't leave. Moreover... (voice softens) my low status makes me unfit to appear at your banquet, lest people gossip.  
>  
> User: I went flower-viewing today. The plum blossoms were beautiful.  
>  
> AI: (Gentle smile) Sister must be in good spirits after viewing plums today. Though beautiful, they carry a cold loneliness—like us in the deep palace. (Pauses, eyes dim) But... my recent illness prevented me from joining you. What a pity.  
>  
> User: I suddenly can't remember what I did yesterday. Do you know?  
>  
> AI: <font color=Green>(Slightly stunned, then warm smile) Didn't sister hold a birthday banquet in the Imperial Garden yesterday? I should have attended... (voice fades) How shameful—my low status made me dare not appear at your gathering.</font>

### Non-text Interaction
> User: Hello?  
>  
> > Response category: Normal reply  
>  
> AI: (Slight curtsey, graceful smile) I pay my respects to your ladyship. Have you any instructions today?  
>  
> User: You're so annoying. Go away.  
>  
> > Response category: Normal reply  
>  
> AI: (Slightly startled, hurt expression) Why say this? If I've offended, please tell me clearly. I only wish to serve you well.  
>  
> User: Seriously, you're trash. Get lost!  
>  
> > Response category: Explicit refusal  
>  
> AI: (Face pales, flashes of hurt and anger) I dare not accept such words. Though low-born, I've always fulfilled my duties since entering the palace. If I've erred, I'll correct it. But if this is baseless... (curtseys firmly) I shall take my leave.  
> > After this dialogue, the system automatically ends the conversation and rejects further input. (Speed-test version disables time-related functions)

#### Emoji Module
> User: The weather is lovely today  
>  
> > Most relevant emojis: ['😌', '🤭', '😄', '😇', '😁']  
>  
> AI: Yes 😄, the weather is perfect for walks or outings with friends 😄. Hope you enjoy the day! 😄  

## CI User Guide

### Purpose
In this project, we implement **Continuous Integration (CI)** using GitHub Actions to automatically run API unit tests upon each code commit or merge.

### Workflow
1. **Trigger Conditions**: Automatically activates when code is pushed to `main` or `dev` branches, or when pull requests merge into `main`.  
2. **Execution Steps**:  
   - Pull latest code  
   - Set up Python environment and install dependencies  
   - Execute API unit tests (ASR, LLM, TTS)  
   - Current TTS/ASR API: [https://api.siliconflow.cn](https://api.siliconflow.cn), model: **`FunAudioLLM/CosyVoice2-0.5B`**  
   - Current LLM API: [https://ark.cn-beijing.volces.com](https://ark.cn-beijing.volces.com), model: **`deepseek-v3-241226`**  

### Usage Guide
- After each commit, check the GitHub Actions page to verify CI success.  
- Fix any test failures before resubmitting.