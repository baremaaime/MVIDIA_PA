from PIL import Image
import os
import re
import glob
import random

import numpy as np
import cv2
import os


folder_path = "training"

video_id_pattern = re.compile(r'^(\d+)\..*\.jpg$')

for category in range(1, 33):
    video_ids = []
    train_dir = f'root/train/{category}/' 
    test_dir = f'root/test/{category}/'
    
    for filename in os.listdir(os.path.join(folder_path, str(category))):
        if filename.endswith(".jpg"):
            match = video_id_pattern.match(filename)
            if match:
                video_id = match.group(1)
                video_ids.append(video_id)

    video_ids = list(set(video_ids))
    random.shuffle(video_ids)

    num_videos = len(video_ids)
    split_index = int(0.15 * num_videos)

    test_videos = video_ids[:split_index]
    train_videos = video_ids[split_index:]

    j = 1
    for id in test_videos:
        pattern = os.path.join(folder_path, str(category), id + ".*.jpg")
        jpg_files = glob.glob(pattern)
       
        image = Image.open(jpg_files[0])
        image.save(test_dir+f'image{j}.jpg')
        j += 1

    j = 1
    for id in train_videos:
        pattern = os.path.join(folder_path, str(category), id + ".*.jpg")
        jpg_files = glob.glob(pattern)

        image = Image.open(jpg_files[0])
        image.save(train_dir+f'image{j}.jpg')
        j += 1