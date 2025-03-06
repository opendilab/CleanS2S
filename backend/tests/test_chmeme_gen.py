import csv
import base64
import os
from openai import OpenAI
import random
import json
import re
import numpy as np
from PIL import Image
# CSV file path
#csv_file_path='/mnt/afs/xueyingyi/meme/data/label_C.csv'
#csv_file_path = "/mnt/afs/niuyazhe/data/meme/data/label_C.csv"
translation = {
    'sentiment category': '情感类别',
    'happiness': '开心',
    'love': '爱',
    'anger': '愤怒',
    'sorrow': '悲伤',
    'fear': '恐惧',
    'hate': '恨',
    'surprise': '惊讶',
    'sentiment degree': '情感程度',
    'slightly': '轻微',
    'moderately': '中等',
    'very': '强烈',
    'intention detection': '情感意图',
    'interactive': '交互',
    'expressive': '表达',
    'entertaining': '娱乐',
    'offensive': '冒犯',
    'other': '其他',
    'offensiveness detection': '冒犯性检测',
    'non-offensive': '非冒犯',
}


def get_tot_data(data_path: str) -> dict:
    # Create a dictionary to store the processed data
    data_dict = {}

    # Open the CSV file and read its contents
    with open(data_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        # Iterate over each row in the CSV
        for row in csv_reader:
            # Use 'images_name' as the key for the outer dictionary
            image_name = row['images_name']
            # Remove 'images_name' from the row to create the inner dictionary
            row.pop('images_name')
            # Store the inner dictionary in the outer dictionary
            data_dict[image_name] = row
    labels = ['sentiment category', 'sentiment degree', 'intention detection', 'offensiveness detection']
    # polish content of the dict
    for image_name, details in data_dict.items():
        for key in labels:
            content = details[key]
            details[key] = content[2:-1]
    #data_dict: image_path -> content dict
    return data_dict


def get_smaples_random(data: dict, sample_num: int = None) -> list[int]:
    #return the index of selected samples
    data_dict = data
    if sample_num is None:
        sample_num = 100
    num_samples = sample_num
    random_numbers = random.sample(range(0, len(data_dict)), num_samples)
    return random_numbers


def encode_image(image_path: str) -> tuple:
    # return base64 image and origin format of the image
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    with Image.open(image_path) as img:
        img_format = img.format.lower()
    return img_base64, img_format


def get_ans_chmeme_ch(
        data_dict: dict, image_list: list, client, end_point: str, cared_keys: list, sys_prompt: str
) -> dict:
    # main function of using api to generate data
    # return: dict, image_num -> detail_dict
    ch_dataset = {}
    error_imgs = []
    token_usage = 0
    for image_path in image_list:
        # set the max number of tiles in `max_num`
        print(f"processing {image_path}...")
        base64_image, image_type = encode_image(image_path)
        img_name = image_path.split('/')[-1]
        # single-image single-round conversation (单图单轮对话)
        #prompt_path='/mnt/afs/wangqijian/meme/ch_memes/prompt_gen_test_chmemes.txt'
        #prompt_path='/mnt/afs/wangqijian/meme/ch_memes/prompt_gen_test_emodata_en.txt'
        img_details = data_dict[img_name]
        question = '\n下面我将根据上面的要求分析这个表情包,并且按照样例的格式输出我的分析结果. 这个表情包的'
        for element in cared_keys:
            detail = f'{translation[element]} 是 {translation[img_details[element]]}, '
            question += detail
        if img_details['metaphor occurrence'] == '1':
            add_prompt = '注意在这个表情包中存在隐喻,其中'
            src = img_details['source domain']
            tgt = img_details['target domain']
            src_parts = re.split(r'[;；]', src)
            tgt_parts = re.split(r'[;；]', tgt)
            for src_item, tgt_item in zip(src_parts, tgt_parts):
                add_prompt += f', \'{tgt_item}\' 有 \'{src_item}\' 的意思'
            question += add_prompt

        try:
            response = client.chat.completions.create(
                # change accroding to your config
                model=end_point,
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt
                    }, {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            ret_text = response.choices[0].message.content
            token_usage += response.usage.total_tokens
            tmpdata = {}
            tmpdata['description'] = ret_text
            for key in cared_keys:
                tmpdata[key] = img_details[key]
            tmpdata['prompt'] = question
            ch_dataset[image_path] = tmpdata
            print(f"processing {image_path} finished")
        #except ZeroDivisionError as e:
        except:
            error_imgs.append(image_path)
            print(f"Error processing image {image_path}")
            #print(e)python
            continue
    print(f'You have used {token_usage} tokens for the dataset generation')
    print(f'The number of error images:{len(error_imgs)}')
    print(f'error imgs:')
    for img in error_imgs:
        print(img)
    return ch_dataset


def save_to_json(data: dict, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_dataset(data: dict, prompt_path: str, client_endpoint: str, prefix: str, select_numbers: list = None) -> dict:
    #data_dict=get_tot_data(csv_file_path)
    data_dict = data
    if select_numbers is None:
        random_numbers = get_smaples_random(data_dict, 10)
    else:
        random_numbers = select_numbers

    random_numbers.sort()
    image_list = []

    #img_l=[656,1371,1558,1584,656,908,186,1037,4410,5554,6004,5976,5609,4299]
    #can be modified
    img_l = random_numbers

    for num in img_l:
        img_path = f'{prefix}({num}).jpg'
        image_list.append(img_path)

    client = OpenAI(
        # get API KEY from env
        api_key=api_key,
        base_url=model_url,
        #base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    cared_keys = ['sentiment category', 'sentiment degree', 'intention detection', 'offensiveness detection']

    with open(prompt_path, 'r', encoding='utf-8') as file:
        PROMPT = file.read()
    ch_dataset = get_ans_chmeme_ch(data_dict, image_list, client, client_endpoint, cared_keys, PROMPT)

    #image_list=['/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(272).jpg']
    json_dataset = {}
    for num in img_l:
        img_path = f'{prefix}({num}).jpg'
        new_img_path = f'{num}'
        cur_data = {}
        cur_details = ch_dataset[img_path]
        text_details = cur_details['description']
        lines = text_details.strip().split('\n')
        for line in lines:
            if line.startswith("(0)"):
                cur_data["content"] = line[3:].strip()

            elif line.startswith("(1)"):
                cur_data["emotion"] = line[3:].strip()

            elif line.startswith("(2)"):
                cur_data["intention"] = line[3:].strip()

            elif line.startswith("(3)"):
                cur_data["scene"] = line[3:].strip()
        json_dataset[new_img_path] = cur_data

    return json_dataset


#csv_file_path='/mnt/afs/wangqijian/meme/ch_memes/label_C.csv'
csv_file_path = os.environ.get("csv_file_path")
#path to the image files
#image_prefix='/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_'
image_prefix = os.environ.get("img_prefix")
#path to the prompt
#prompt_path='/mnt/afs/wangqijian/meme/ch_memes/prompt_gen_test_chmemes_ch.txt'
prompt_path = os.environ.get("prompt_path")
#path to save the dataset
#dataset_path_json='/mnt/afs/wangqijian/meme/ch_memes/dataset_4.json'
dataset_path_json = os.environ.get("data_path_json")
#new_json_path='/mnt/afs/wangqijian/meme/ch_memes/test.json'
#need to set API_KEY
api_key = os.environ.get("ARK_API_KEY")
#set model_url
model_url = "https://ark.cn-beijing.volces.com/api/v3"
#first need to set model_endpoint in website
model_endpoint = "ep-20250124115350-pwjxt"


def main():
    data = get_tot_data(csv_file_path)
    processed_numbers = [
        272, 4123, 3686, 497, 4997, 5405, 5695, 1879, 262, 1747, 1998, 1869, 2477, 654, 1261, 5335, 3862, 3498, 5629,
        2968, 1873, 1185, 5677, 1765, 1217, 5253, 3787, 2489, 5645, 5567, 3715, 2209, 2622, 81, 1401, 1489, 835, 1515,
        1328, 2862, 367, 3306, 671, 1095, 4732, 3097, 459, 5485, 6014, 427, 853, 82, 1410, 5705, 305, 4033, 989, 522,
        462, 5452, 4147, 1098, 3999, 3240, 705, 1058, 1110, 1812, 4249, 4128, 2720, 735, 2972, 1864, 2370, 5380, 4488,
        4730, 908, 3923, 4884, 5014, 3096, 2463, 3015, 4947, 931, 4984, 5200, 3626, 6012, 3315, 628, 3734, 708, 2805,
        5022, 5552, 5765, 1072
    ]
    #tot_numbers=range(600,1800)
    #tot_numbers=range(1800,4200)
    tot_numbers = range(0, 6046)
    p_np = np.array(processed_numbers)
    t_np = np.array(tot_numbers)
    select_numbers = list(np.setdiff1d(t_np, p_np))
    json_dataset = get_dataset(data, prompt_path, model_endpoint, image_prefix, select_numbers)
    save_to_json(json_dataset, dataset_path_json)


main()
