from base import BaseDataSet, BaseDataLoader
from PIL import Image
from glob import glob
import numpy as np
from utils import palette
import torch
import os
import cv2
import json
import pandas as pd

class Imaterialist(BaseDataSet):
    def __init__(self, warp_image = True, **kwargs):
        
        self.DATA_DIR = os.path.join("data","imaterialist")
        with open(os.path.join(self.DATA_DIR, "label_descriptions.json")) as f:
            self.label_descriptions = json.load(f)
        
        segment_df = pd.read_csv(os.path.join(self.DATA_DIR,"train.csv"))
        label_names = [x['name'] for x in self.label_descriptions['categories']]
        
        self.image_df = self._get_image_df(segment_df)
        self.warp_image = warp_image
        self.num_classes = len(label_names)
        self.palette = palette.get_voc_palette(self.num_classes)
        self.file_index = list(self.image_df.index)
        super(Imaterialist, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ['train', 'test']:
            file_path = os.path.join(self.DATA_DIR, self.split, "*")
            self.files = [os.path.basename(x) for x in glob(file_path)]
        else: raise ValueError(f"Invalid split name {self.split} choose one of [train, test]")

    def _load_data(self, index):
        image_id = self.file_index[index]
        image_path = os.path.join(self.DATA_DIR, self.split, image_id + ".jpg")
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        
        # get label from rle
        label = self._get_label(f = image_id, shape = image.shape)
        
        if self.warp_image:
            image = cv2.resize(image, (513, 513), interpolation=cv2.INTER_AREA)
            label = np.asarray(Image.fromarray(label).resize((513, 513), resample=Image.NEAREST))
        return image, label, image_id
    
    def _get_image_df(self, segment_df):
        image_df = segment_df.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))
        size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
        image_df = image_df.join(size_df, on='ImageId')
        return image_df

    def _get_label(self, f, shape):
        encoded_pixels = self.image_df.loc[f, 'EncodedPixels']
        class_ids = self.image_df.loc[f, "ClassId"]
        height, width = shape[:2]
        
        mask = np.zeros((height, width)).reshape(-1)
        for segment, (pixel_str, class_id) in enumerate(zip(encoded_pixels, class_ids)):
            splitted_pixels = list(map(int, pixel_str.split()))
            pixel_starts = splitted_pixels[::2]
            run_lengths = splitted_pixels[1::2]
            assert max(pixel_starts) < mask.shape[0]
            for pixel_start, run_length in zip(pixel_starts, run_lengths):
                pixel_start = int(pixel_start) - 1
                run_length = int(run_length)
                mask[pixel_start:pixel_start+run_length] = 255 - class_id * 4
        mask = mask.reshape([height, width], order='F')
        return mask
        
def get_parent_class(value, dictionary):
    for k, v in dictionary.items():
        if isinstance(v, list):
            if value in v:
                yield k
        elif isinstance(v, dict):
            if value in list(v.keys()):
                yield k
            else:
                for res in get_parent_class(value, v):
                    yield res

class IMAT(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, val=False):

        self.MEAN = [0.43931922, 0.41310471, 0.37480941]
        self.STD = [0.24272706, 0.23649098, 0.23429529]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = Imaterialist(**kwargs)

        super(IMAT, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

