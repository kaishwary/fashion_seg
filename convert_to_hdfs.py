from dataloaders import IMAT
from dataloaders.imat_red import IMATREDLoader
import torch
import h5py
import time
import os
import numpy as np
from pathlib import Path
from PIL import Image
#%% Create path if not exists

Path(os.path.join("data", "imat_reduced", "train")).mkdir(parents=True, exist_ok=True)
Path(os.path.join("data", "imat_reduced", "test")).mkdir(parents=True, exist_ok=True)
Path(os.path.join("data", "imat_reduced", "new")).mkdir(parents=True, exist_ok=True)
#%%

def store_single_hdf5(image, label, image_id):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(os.path.join("data", "imat_reduced", "train", f"{image_id}.h5"), "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()

def convert_to_hdfs(image, mask, image_id):
    store_single_hdf5(image, mask, image_id)
    
def converter_func(start_idx, end_idx, loader):
    print("Converting from", start_idx, "to", end_idx)
    start = time.time()
    for i in range(start_idx, end_idx):
        image, mask, image_id = loader.dataset._load_data(i)
        convert_to_hdfs(image, mask, image_id)
    end = time.time()
    print("Time taken", end-start)
    
def converter_jpeg(loader):
#    print("Converting from", start_idx, "to", end_idx)
    start = time.time()
    for i in range(20):
        image, mask, image_id = loader.dataset._load_data(i)
        np.save(arr = image,file = os.path.join("data", "imat_reduced", "new", image_id))
    end = time.time()

    print("Time taken", end-start)

#%%   
if __name__ == "__main__":
    
    start = time.time()
    loader = IMAT(data_dir="af", num_workers=0, batch_size=10, split="train")
    
    import sys

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    
    converter_func(start_idx, end_idx, loader)
    
    end = time.time()
    print("Total time taken:", end-start)