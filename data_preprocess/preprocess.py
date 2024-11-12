import os
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
from PIL import Image
import numpy as np

# set your paths here
input_label_dir = Path('./data_preprocess/labels')
output_dir = Path('./datasets/MyDataset')
image_dir = Path('./data_preprocess/images')

# create output directories if don't exist
output_dir.mkdir(parents=True, exist_ok=True)

label_file_list = os.listdir(input_label_dir)

image_suffix = next(image_dir.iterdir()).suffix

# 读取预先设定的分词用的词汇
with open('./data_preprocess/vocab.txt', 'r', encoding='utf-8') as f:
    vocab = f.read().split()

def FMM_func(
    vocab_list: list[str], 
    sentence: str
):
    """
    正向最大匹配（FMM）
    :param user_dict: 词典
    :param sentence: 句子
    """
    # 词典中最长词长度
    max_len = len(max(vocab_list, key=len))
    start = 0
    token_list = []
    while start != len(sentence):
        index = start + max_len
        if index > len(sentence):
            index = len(sentence)
        for _ in range(max_len):
            token = sentence[start:index]
            if (token in vocab_list) or (len(token) == 1):
                token_list.append(token)
                # print(token, end='/')
                start = index
                break
            index -= 1
    return token_list

def process_file(file_name):
    ID = os.path.splitext(file_name)[0]

    empty_dict = {
        "ID": ID,
        "label": ""
    }
    # skip missing image
    if not os.path.exists(f"{image_dir}/{ID}{image_suffix}"):
        return empty_dict

    with open(input_label_dir / file_name, 'r', encoding='utf-8') as f:
        content = f.read()

    token_list = FMM_func(vocab, content)
    token_list = [token_list[i] for i in range(len(token_list)) if token_list[i] != ' ']  # Remove spaces

    new_content = ' '.join(token_list)

    # ignore non-standard tokens
    for token in token_list:
        if token not in vocab and token not in ['', ' ']:
            return empty_dict

    # ignore error mathpix and multi-line labels
    lines = new_content.splitlines()
    if 'error mathpix' in content or len(lines) > 1: 
        return empty_dict

    # correct dict
    return {
        "ID": ID,
        "label": new_content,
        "image": np.array(Image.open(f"{image_dir}/{ID}{image_suffix}"))
    }
    

if __name__ == '__main__':
    with Pool() as pool:
        dict_list = pool.imap_unordered(process_file, label_file_list, chunksize=10)
        dict_list_final = []
        counter_removed = 0
        counter_total = 0
        counter_batch = 0
        for label_image_dictionary in tqdm(dict_list, total=len(label_file_list)):
            counter_total += 1
            if label_image_dictionary["label"] != "":
                dict_list_final.append(label_image_dictionary)
                # Write to file after a certain number of instances to prevent out of memory
                if len(dict_list_final) % 10000 == 0:
                    print(f"Writing part of data to {output_dir}/batch_{counter_batch}.npy")
                    np.save(f'{output_dir}/batch_{counter_batch}.npy', dict_list_final, allow_pickle=True)
                    counter_batch += 1
                    dict_list_final = []
            else:
                counter_removed += 1

        if len(dict_list_final) > 0:
            np.save(f'{output_dir}/batch_{counter_batch}.npy', dict_list_final, allow_pickle=True)

        print(
            f"""
            Summary:
            - Original number of labels: {counter_total}
            - Removed number of labels: {counter_removed}
            - Output number of labels: {counter_total - counter_removed}
            - Output path: {output_dir}
            """
        )

        


