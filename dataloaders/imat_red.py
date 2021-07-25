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
import h5py

class ImatRed(BaseDataSet):
    def __init__(self, warp_image = True, **kwargs):
        self.warp_image = warp_image
        self.num_classes = 46
        self.palette = palette.get_voc_palette(self.num_classes)
        self.root = os.path.join("data","imat_reduced")
        super(ImatRed, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ['train', 'test']:
            file_path = os.path.join(self.root, self.split, "*")
            self.files = [os.path.basename(x).split(".")[0] for x in glob(file_path)]
        else: raise ValueError(f"Invalid split name {self.split} choose one of [train, test]")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.root, self.split, image_id + ".h5")
        # load data from h5
        image, label = self._read_single_hdf5(image_path)
        return image, label, image_id
    
    @staticmethod
    def _read_single_hdf5(file_path):
        # Open the HDF5 file
        file = h5py.File(file_path, "r+")
    
        image = np.array(file["/image"]).astype("uint8")
        label = np.array(file["/meta"]).astype("uint8")
    
        return image, label
    
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

class IMATREDLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, val=False):

        self.MEAN = [0.3711, 0.3283, 0.4485]
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

        self.dataset = ImatRed(**kwargs)

        super(IMATREDLoader, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

