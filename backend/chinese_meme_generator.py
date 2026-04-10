import csv
import base64
import os
from openai import OpenAI
import random
import json
import re
import numpy as np
from PIL import Image

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


def get_total_data(data_path: str) -> dict:
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
            # original data: 4(sorrow)/2(moderately)
            # using content[2:-1] to remove number & '()'
            details[key] = content[2:-1]
    #data_dict: image_path -> content dict
    return data_dict


def get_samples_random(data: dict, sample_num: int = None) -> list[int]:
    #return the index of selected samples
    if sample_num is None:
        sample_num = 100
    random_numbers = random.sample(range(0, len(data)), sample_num)
    return random_numbers


def encode_image(image_path: str) -> tuple:
    # return base64 image and origin format of the image
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    with Image.open(image_path) as img:
        img_format = img.format.lower()
    return img_base64, img_format


def get_ans_chmeme_chinese(
        data_dict: dict, image_list: list, client, end_point: str, cared_keys: list, sys_prompt: str
) -> dict:
    # main function of using api to generate data
    # return: dict, image_num -> detail_dict
    ch_dataset = {}
    error_imgs = []
    token_usage = 0
    for image_path in image_list:
        print(f"processing {image_path}...")
        base64_image, image_type = encode_image(image_path)
        img_name = image_path.split('/')[-1]
        img_details = data_dict[img_name]
        question = '\n下面我将根据上面的要求分析这个表情包，并且按照样例的格式输出我的分析结果。这个表情包的'
        for element in cared_keys:
            detail = f'{translation[element]} 是 {translation[img_details[element]]}, '
            question += detail
        if img_details['metaphor occurrence'] == '1':
            add_prompt = '注意在这个表情包中存在隐喻，其中'
            src = img_details['source domain']
            tgt = img_details['target domain']
            src_parts = re.split(r'[;；]', src)
            tgt_parts = re.split(r'[;；]', tgt)
            for src_item, tgt_item in zip(src_parts, tgt_parts):
                add_prompt += f"， ‘{tgt_item}’ 有 ‘{src_item}’ 的意思"
            add_prompt += '。'
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
        except:
            error_imgs.append(image_path)
            print(f"Error processing image {image_path}")
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


def get_dataset(
        data: dict, prompt_path: str, client_endpoint: str, img_prefix_dir: str, select_numbers: list = None
) -> dict:
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
        img_path = os.path.join(img_prefix_dir, f"Image_({num}).jpg")
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
    ch_dataset = get_ans_chmeme_chinese(data_dict, image_list, client, client_endpoint, cared_keys, PROMPT)

    json_dataset = {}
    for num in img_l:
        img_path = f'{img_prefix_dir}({num}).jpg'
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


csv_file_path = os.environ.get("csv_file_path")
# path to the image files
image_prefix = os.environ.get("img_prefix")
# path to the prompt
prompt_path = './chinese_meme_generation_prompt.txt'
# path to save the dataset
dataset_path_json = os.environ.get("data_path_json")
# need to set API_KEY
api_key = os.environ.get("ARK_API_KEY")
# set model_url
model_url = "https://ark.cn-beijing.volces.com/api/v3"
# first need to set model_endpoint in website
model_endpoint = "ep-20250124115350-pwjxt"


def main():
    data = get_total_data(csv_file_path)
    select_numbers = range(0, 6046)
    json_dataset = get_dataset(data, prompt_path, model_endpoint, image_prefix, select_numbers)
    save_to_json(json_dataset, dataset_path_json)


if __name__ == "__main__":
    main()
