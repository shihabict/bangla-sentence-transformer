import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from trianing.tarin import BNSentenceTransformer

if __name__ == '__main__':
    transformer = BNSentenceTransformer()

    # path = 'DATA/dataset.txt'
    #path = 'DATA/hf_aibarat_dataset.txt'
    path = 'DATA/sample_dataset.txt'
    number_of_sentences = 'Full_data'
    #number_of_sentences = 100
    save_model_name = f'bangla_transformer_{number_of_sentences}'
    world_size = torch.cuda.device_count()
    mp.spawn(transformer.train_new, args=(world_size, path,number_of_sentences,save_model_name), nprocs=world_size)
    # transformer.train_new(path,number_of_sentences,save_model_name)

