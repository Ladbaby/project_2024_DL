import os
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
from PIL import Image
import numpy as np

# set your paths here
input_label_dir = Path('./data_preprocess/labels')
output_dir = Path('./datasets/MyDataset_test')
image_dir = Path('./data_preprocess/images')

# create output directories if don't exist
output_dir.mkdir(parents=True, exist_ok=True)

label_file_list = os.listdir(input_label_dir)
image_file_list = os.listdir(image_dir)

# Create a set of label file names (without extensions if necessary)
label_set = set(os.path.splitext(label)[0] for label in label_file_list)

# Find images with missing labels
missing_label_images = [image for image in image_file_list if os.path.splitext(image)[0] not in label_set]

image_suffix = next(image_dir.iterdir()).suffix

def process_file(file_name):
    ID = os.path.splitext(file_name)[0]

    empty_dict = {
        "ID": ID,
        "label": ""
    }
    # skip missing image
    if not os.path.exists(f"{image_dir}/{ID}{image_suffix}"):
        return empty_dict

    # correct dict
    return {
        "ID": ID,
        "label": "NOT_AVAILABLE",
        "image": np.array(Image.open(f"{image_dir}/{ID}{image_suffix}"))
    }
    

if __name__ == '__main__':
    with Pool() as pool:
        dict_list = pool.imap_unordered(process_file, missing_label_images, chunksize=2000) # lazy way to prevent ID order changed by multi-processing
        dict_list_final = []
        counter_removed = 0
        counter_total = 0
        counter_batch = 0
        for label_image_dictionary in tqdm(dict_list, total=len(missing_label_images)):
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

        


