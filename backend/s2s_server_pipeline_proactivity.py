from typing import Dict, List, Optional, Union, Any
import json
import os
import re
import logging
from openai import OpenAI
from s2s_server_pipeline import LanguageModelAPIHandler
import torch
from sklearn.metrics.pairwise import cosine_similarity
from promcse import PromCSE
from FlagEmbedding import FlagAutoModel
from transformers import AutoModel
from datasets import load_dataset
from enum import Enum
# Configure logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

folder_path = "logging"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)


file_handler = logging.FileHandler('logging/memory_logger.txt')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(file_handler)
logger.propagate = False


class Proactivity:
    # Memory saves the facts，elements and history messages in conversation，history_len is used to decide the length of history messages. When the length of history messages is longer than history_len,
    # the former messages will be processed and saved by summary. After every turn of conversation, self.facts and self.elements will be updated.

    def __init__(self, history_len, model_url, model_name, embedding_model_name='jina') -> None:
        """
        Arguments:
            - history_len(int): the length of saved message
            - model_url(str): the url of model
            - model_name(str): the name of model
        """
        self.history_len = history_len  # The length of history messages to save
        self.facts = ['' for _ in range(self.history_len)] # list but use as a deque(slice option needed)
        self.summary = []
        self.history_list = ['' for _ in range(self.history_len)]
        self.history_json = "[]"
        # Break down key facts into multiple elements
        self.elements = {'时间': '', '地点': '', '气候': '', '人物': '', '动作': '', '态度': ''}
        self.client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=model_url)
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')
        self.memory_base_path = os.path.join(self.base_path, 'proactivity')
        with open(os.path.join(self.memory_base_path, 'fact.txt'), "r", encoding='utf-8') as f:
            self.fact_sys_prompt = f.read()
        with open(os.path.join(self.memory_base_path, 'summary.txt'), "r", encoding='utf-8') as f:
            self.summary_sys_prompt = f.read()
        with open(os.path.join(self.memory_base_path, "initialize.txt"), "r", encoding='utf-8') as f:
            self.initialize_sys_prompt = f.read()
        with open(os.path.join(self.memory_base_path, "update.txt"), "r", encoding='utf-8') as f:
            self.update_sys_prompt = f.read()
        with open(os.path.join(self.memory_base_path, "inside_conflict.txt"), "r", encoding='utf-8') as f:
            self.conflict_sys_prompt = f.read()
        with open(os.path.join(self.memory_base_path, "reject.txt"), "r", encoding='utf-8') as f:
            self.reject_sys_prompt = f.read()
        with open(os.path.join(self.memory_base_path, "panding.txt"), "r", encoding='utf-8') as f:
            self.panding_sys_prompt = f.read()
        ds = load_dataset("AltmanD/emoji_info")
        self.emoji_dataset = self._csv2json(ds)
        if self.embedding_model_name == 'bert':
            self.embedding_model = PromCSE("hellonlp/promcse-bert-base-zh-v1.1", "cls", 10)
        elif self.embedding_model_name == 'jina':
            self.embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        elif self.embedding_model_name == 'beg':
            self.embedding_model = FlagAutoModel.from_finetuned('BAAI/bge-base-zh-v1.5',
                                            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                            use_fp16=True)
        else:
            raise KeyError(f'Incorrect arg: {self.embedding_model_name}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = self.embedding_model.to(device)

    def call_llm(self, user_messages, system_prompt, temperature=0.6, isjson=False):
        """
        Call the language model API to generate text.
        Arguments:
            - messages(string): user message
            - system_prompt(string): system prompt
            - temperature(float): control the temperature in generation
            - isjson(bool): control flag of json format generation
        Returns:
            - output (str): The generated text (or json format) output.
        """
        msg = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_messages
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=msg,
            max_tokens=4096,
            temperature=temperature,
            stream=False,
            frequency_penalty=0,
            presence_penalty=0,
            top_p=0.95,
            logprobs=False,
            response_format={"type": "json_object"} if isjson else None
        )
        t = response.choices[0].message.content

        if isjson:
            try:
                res = json.loads(t)
            except json.JSONDecodeError:
                res = t
        else:
            res = t

        return res

    def add_2_memory(self, new_msg) -> None:  # add new_msg to self.facts and self.summary
        out_msg = self.history_list.pop(0)
        self.history_list.append(new_msg)
        self._fact_func(new_msg)
        if out_msg:
            self._summary_func(out_msg)

    def get_from_memory(self) -> json:
        fact = json.dumps(self.facts[-self.history_len // 2:], ensure_ascii=False)
        summary = json.dumps(self.summary, ensure_ascii=False)
        hist = json.dumps(self.history_list, ensure_ascii=False)
        elements = json.dumps(self.elements, ensure_ascii=False)
        return json.dumps({"关键事实": fact, "超限对话总结": summary, "关键事实要素": elements, "历史对话": hist}, ensure_ascii=False)

    def get_topk_emoji(self, query_sentence, k=5):
        scorelist = self._scores_match_sentences(query_sentence)
        sorted_with_index = sorted(enumerate(scorelist), key=lambda x: x[1], reverse=True)

        top_k_indexes = [index for index, _ in sorted_with_index[:k]]
        
        res = []
        keylist = list(self.emoji_dataset.keys())
        for k in top_k_indexes:
            res.append(keylist[k])
        return res

    def _scores_match_sentences(self, query_sentence):

        ranklist = list()
        sentences = list()

        for indexs, emojis in self.emoji_dataset.items():
            ranklist.append(emojis[6::3])
            for i in range(4, len(emojis), 3):
                sentences.extend(emojis[i:i+2])
        
        max_rank = max([max(ranks) for ranks in ranklist])
        simlist = self._cal_sim(query_sentence, sentences)
        scorelist = list()

        simindex = 0
        for rank in ranklist:
            if len(rank) != 0:
                tscore = sum([(0.6 * simlist[i+simindex] + 0.4 * simlist[i+simindex+1]) * (rank[i] / max_rank) for i in range(len(rank))]) / (len(rank) ** 0.5)
                simindex += len(rank)
                scorelist.append(tscore)
            else:
                scorelist.append(0)
        return scorelist

    def _cal_sim(self, query_sentence, sentences):
        if self.embedding_model_name == 'bert':
            similarities = self.embedding_model.similarity(query_sentence, sentences)
        else:
            embeddings = self.embedding_model.encode([query_sentence] + sentences)
            embeddings_query = embeddings[0].reshape(1, -1)
            embeddings_sentences = embeddings[1:]
            similarities = cosine_similarity(embeddings_query, embeddings_sentences)
        return similarities[0]

    def _fact_func(self, msg) -> None:
        old_fact = self.facts[:-self.history_len // 2]
        self.facts.pop(0)
        self.facts.append(self.call_llm("对话：" + msg + "旧的关键事实：" + str(old_fact), self.fact_sys_prompt))
        logger.info(f"Memory fact updated: {self.facts}")

    def _summary_func(self, msg) -> None:
        old_summary = str(self.summary)
        self.summary = self.call_llm("对话：" + msg + "旧的总结：" + old_summary, self.summary_sys_prompt)
        logger.info(f"Memory summary updated: {self.summary}")

    def _update(
            self, update_elements_list, msg
    ) -> None:  # update self.elements according to the result of self.panding()
        for update_element in update_elements_list:
            if self.elements[update_element] == '':  # not initialized yet
                umsg = f"对话：{msg}, 要素：{update_element}"
                initialization = self.call_llm(umsg, self.initialize_sys_prompt)
                self.elements[update_element] = initialization
                logger.info(f"Memory element {update_element} initialized: {initialization}")
            else:
                umsg = f"对话：{msg}, 要素：{update_element}, 已有要素内容：{self.elements[update_element]}"
                updated = self.call_llm(umsg, self.update_sys_prompt)
                self.elements[update_element] += updated
                logger.info(f"Memory element {update_element} updated: {updated}")

    def _inside_conflict(
        self, msg, element
    ) -> bool:  #Judge whether there is a conflict between QA and element. Refer only to the content in self.elements

        umsg = f'对话：{msg}, 要素：{element}, 要素内容：{self.elements[element]}'
        reject_flag = self.call_llm(umsg, self.conflict_sys_prompt)
        if 'True' in reject_flag:
            return True
        else:
            return False

    def _reject(self, msg) -> {bool, list}:  # Refer only to the content in self.elements
        reject_list = []
        for element in self.elements.keys():
            if self.elements[element] != '':  # already initialized
                conflict_flag = self.inside_conflict(msg, element)
                if conflict_flag:
                    umsg = f'对话：{msg}, 要素：{element}, 要素内容：{self.elements[element]}'
                    reject = self.call_llm(umsg, self.reject_sys_prompt)
                    reject_list.append({'要素': element, '原因': reject})
        if len(reject_list) > 0:
            return True, reject_list
        return False, ''

    def _panding(self, msg) -> list:  # decide which elements to be considered
        umsg = f"用户输入：{msg},关键事实种类列表：{self.elements.keys()}"
        elements_list = self.call_llm(umsg, self.panding_sys_prompt, isjson=True)
        if isinstance(elements_list, dict):
            first_key = list(elements_list.keys())[0]
            elements_list = elements_list[first_key]
        return elements_list

    
    def _csv2json(self, dataset) -> dict:  # decide which elements to be considered
        res = {}
        for x in dataset['test']:
            tlist = list()
            xlist = list(x.items())
            for i, (k, v) in enumerate(xlist):
                if k != 'emoji':
                    if v != '-':
                        match = re.match(r"Q\d{1,2}(?:0[1-9]|100)?rank", k)
                        if match:
                            tlist.append(int(v))
                        else:
                            tlist.append(v)
                    if v == '-' or i == len(xlist)-1:
                        res[x['emoji']] = tlist
                        break
        return res

    def process(self, msg) -> {bool, str}:  # finally process msg, reject or update
        reject_flag, reject_result = self._reject(msg)
        if reject_flag:  #If there is a conflict
            return False, reject_result
        else:
            elements_list = self._panding(msg)
            self._update(elements_list, msg)
            return True, ''

    def clear(self) -> None:
        self.facts = []
        self.summary = []
        self.history_list = ['' for _ in range(self.history_len)]


class ChatMode(Enum):
    REGULAR_MODE = 1
    MEMORY_ONLY = 2
    NONTEXT_INTERACTION_ONLY = 3
    EMOJI_ONLY = 4
    VIRTUALCHARACTER_ONLY = 5


class ProactivityChatHelper:
    def __init__(self, model_url, model_name, character='anlingrong.txt', history_len=5, mode=ChatMode.REGULAR_MODE) -> None:
        self.agent = Proactivity(history_len, model_url, model_name)
        self.character_base_path = os.path.join(self.agent.base_path, 'character')
        self.mode = mode
        with open(os.path.join(self.character_base_path, character), "r", encoding='utf-8') as f:
            self.c_sys_prompt = f.read()
        with open(os.path.join(self.agent.memory_base_path, "./nci.txt"), "r", encoding='utf-8') as f:
            self.judge_sys_prompt = f.read()

    def generate_sys_prompt(self, msg):
        # This is an interactive module where the model, based on historical information, can choose from the following actions:
        # ['perfunctory response', 'delayed reply', 'change the subject', 'direct refusal', 'no response', 'normal reply'].
        
        sys_prompt = self.c_sys_prompt
        if self.mode in [ChatMode.REGULAR_MODE, ChatMode.MEMORY_ONLY]:
            sys_prompt += self.agent.get_from_memory()
            # flag means whether the msg is rejected. True means not rejected
            # Not activated in this version
            # flag, reject_result = self.agent.process(msg)
            # sys_prompt += reject_result

        if self.mode in [ChatMode.REGULAR_MODE, ChatMode.NONTEXT_INTERACTION_ONLY, ChatMode.EMOJI_ONLY]:
            his_msg = json.dumps(self.agent.history_list, ensure_ascii=False)
            return_type = self.agent.call_llm('历史对话：' + his_msg + 'user:' + msg, self.judge_sys_prompt)
            judge_type = ['敷衍', '延迟回复', '转移话题', '直白拒绝', '不回复', '正常回复', 'emoji回复']

            # ensure llm responses have correctly processed
            if return_type in ["1", "2", "3", "4", "5", "6", "7"]: # for correct return,1-7 represent type in 'judge_type'
                return_type = int(return_type)
            elif return_type in judge_type: # in case llm replies in Chinese
                return_type = judge_type.index(return_type) + 1
            else: # other cases
                return_type = 6

            # these 2 conditions are not implemented for speed test
            # if re_tyep == 2: # 延迟回复
            # time.sleep(1000)
            # elif re_tyep == 5: # 不回复
            #     res = ''
            # else: # 其他情况

            if return_type in [1, 3, 4] and self.mode != ChatMode.EMOJI_ONLY:
                sys_prompt += f'# 指导思想：此次回复的指导思想为：{judge_type[return_type-1]}'
                print(f'AI 内心：{judge_type[return_type-1]}')
            elif return_type == 2:
                print('AI 选择延迟回复你')
            elif return_type == 5:
                print('AI 不想回复你')
            elif return_type == 7 or self.mode == ChatMode.EMOJI_ONLY:
                res = self.agent.get_topk_emoji(msg)
                sys_prompt += f'# 指导思想：此次回复的指导思想为: emoji回复，可用emoji有{res}'
            else:
                sys_prompt += f'# 指导思想：此次回复的指导思想为：正常回复'
        print('===========================')
        print(sys_prompt)
        print('===========================')
        return sys_prompt

    def add_2_agent(self, msg):
        self.agent.add_2_memory(msg)

    def clear(self):
        self.agent.clear()


class LanguageModelAPIHandlerProactivity(LanguageModelAPIHandler):

    def __init__(self, *args, character='anlingrong.txt', history_len=5, mode=ChatMode.REGULAR_MODE, **kwargs):
        super().__init__(*args, **kwargs)
        self.proactivity_chat_helper = ProactivityChatHelper(self.model_url, self.model_name, character, history_len, mode)

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

        logger.info(f"User input: {inputs['data']}")
        generator = super().process(inputs)
        total_answer = ""
        for ele in generator:
            if isinstance(ele['answer_text'], str):
                total_answer += ele['answer_text']
            yield ele
        logger.info(f"Model output: {total_answer}")
        self.proactivity_chat_helper.add_2_agent(json.dumps({"user": inputs['data'], "AI": total_answer}, ensure_ascii=False))

    def _before_process(self, prompt: str, count: int) -> List[Dict[str, str]]:
        """
        Preparation chat messages before the generation process.
        Arguments:
            - prompt (str): The input prompt in current step.
            - count (int): The user input count.
        Returns:
            - messages (List[Dict[str, str]): The chat messages.
        """

        sys_p = self.proactivity_chat_helper.generate_sys_prompt(prompt)
        logger.info(f"System prompt generated: {sys_p}")
        self.chat.init_chat({"role": 'system', "content": sys_p})
        self.chat.append({"role": self.user_role, "content": prompt})
        return self.chat.to_list()

    def clear_current_state(self):
        super().clear_current_state()
        self.proactivity_chat_helper.clear()
        logger.info("Memory cleared.")
