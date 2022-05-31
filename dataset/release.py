import os
import SharedArray as sa
import h5py
from tqdm import tqdm

bert_file = './data/text_feature/text_seq.h5'
with h5py.File(bert_file, 'r') as fp:
    for key in tqdm(fp.keys()):
        sa.delete("shm://{}".format(key))