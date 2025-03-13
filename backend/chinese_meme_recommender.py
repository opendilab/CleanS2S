import json
import os
import torch
from transformers import AutoModel
import requests
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


class ChineseMemeRecommender:

    def __init__(self, data: dict, embedding_model_name: str = 'online'):
        self.data = data
        self.embedding_model_name = embedding_model_name
        # ids of chmeme
        self.rec_nums = list(data.keys())
        keys = list(data[self.rec_nums[0]].keys())
        self.key_len = len(keys)
        if embedding_model_name == 'online':
            self.embedding_model_url = os.getenv("EMBEDDING_URL")
            if self.embedding_model_url is None:
                self.embedding_model_url = os.environ.get("embedding_url")
        elif embedding_model_name == 'jina':
            self.embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        else:
            raise KeyError(f'Incorrect arg: {self.embedding_model_name}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.embedding_model_name != 'online':
            self.embedding_model = self.embedding_model.to(device)

    def _get_embed_sim(self, query: str):
        data_sentence = []
        for num in self.rec_nums:
            item = self.data[num]
            for key in list(item.keys()):
                data_sentence.append(item[key])
        if self.embedding_model_name == 'bert':
            similarities = self.embedding_model.similarity(query, data_sentence)
        else:
            if self.embedding_model_name == 'online':
                payload = {
                    "text": [query] + data_sentence,
                    "model": "bge-large-zh-v1.5"
                }
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.embedding_model_url, json=payload, headers=headers)
                embeddings = response.json()['embedding']
            else:
                embeddings = self.embedding_model.encode([query] + data_sentence)
            embeddings_query = torch.tensor(embeddings[0]).reshape(1, -1)
            embeddings_sentences = torch.tensor(embeddings[1:])
            similarities = cosine_similarity(embeddings_query, embeddings_sentences)
        return similarities[0]

    def _get_sim_score(self, query: str):
        sims = self._get_embed_sim(query)
        sims = torch.from_numpy(sims)
        key_len = self.key_len
        sims = sims.view(-1, key_len).float()
        scores = sims.mean(dim=1).tolist()
        return scores

    def get_topk_meme(self, query: str, k: int = 2) -> list:
        scores = self._get_sim_score(query)
        sorted_with_index = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        top_k_indexes = [index for index, _ in sorted_with_index[:k]]

        res = []
        for ind in top_k_indexes:
            res.append(self.rec_nums[ind])
        return res


def test_rec_chmeme(file_path:str, query: str = '耗子尾汁', k: int = 4):
    # json file path
    with open(file_path, 'r', encoding='utf-8') as file:
        ch_dataset = json.load(file)
    # as for ChineseMemeRecommender, only support online embedding model
    # other models can be added later
    my_chmeme = ChineseMemeRecommender(data=ch_dataset)
    return_id = my_chmeme.get_topk_meme(query, k)
    prefix = os.environ.get('prefix')
    image_path = [f'{prefix}({x}).jpg' for x in return_id]
    for path in image_path:
        print(path)
        img = Image.open(path)
        img.show()

if __name__ == "__main__":
    file_path = ''  #your file path
    test_rec_chmeme(file_path)
