#!/usr/bin/env python
# coding: utf-8

# In[1]:




# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import matplotlib.pyplot as plt
from os import path
import os

from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.optim import AdamW, Adam
from torch import autocast, GradScaler

from omegaconf import OmegaConf, open_dict
from utils import save_results

from datetime import datetime

import json

import torch


def print_available_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        # print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            # print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            gpu_memory = torch.cuda.memory_reserved(i)
            gpu_total_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_free_memory = gpu_total_memory - gpu_memory
            # print(f"GPU {i} Free Memory: {gpu_free_memory / 1024**3:.2f} GB")
    else:
        print("No GPUs available")


print_available_gpus()

# In[9]:


# additional packages
# pip install hiera-transformer
# pip install -U pytorch_warmup


# # Hyperparameters

# In[3]:

def train_model(paths, output_file, outer_config, inner_index):

    print(f"Training model with paths: {paths}")
    print(f"GPU ID: {inner_index}")



    gpu_id = inner_index
    device = f'cuda:{gpu_id}'
    
    video_size = [36, 64]
    batchsize=outer_config['batchsize']

    screen_chunk_size = 30
    screen_sampling_rate = 30

    response_chunk_size = 8
    response_sampling_rate = 8

    behavior_as_channels = True
    replace_nans_with_means = True

    dim_head = 64
    num_heads = 2
    drop_path_rate = 0
    mlp_ratio=4


    # ### get dataloaders

    # In[4]:


    full_paths = [path.join("/data/mouse_polly/", f) for f in paths]


    # In[5]:


    from experanto.dataloaders import get_multisession_dataloader
    from experanto.configs import DEFAULT_CONFIG as cfg


    # In[6]:


    cfg.dataset.global_chunk_size = None
    cfg.dataset.global_sampling_rate = None

    cfg.dataset.modality_config.screen.chunk_size = screen_chunk_size
    cfg.dataset.modality_config.screen.sampling_rate = screen_sampling_rate
    cfg.dataset.modality_config.eye_tracker.chunk_size = screen_chunk_size
    cfg.dataset.modality_config.eye_tracker.sampling_rate = screen_sampling_rate
    cfg.dataset.modality_config.treadmill.chunk_size = screen_chunk_size
    cfg.dataset.modality_config.treadmill.sampling_rate = screen_sampling_rate

    cfg.dataset.modality_config.responses.chunk_size = response_chunk_size
    cfg.dataset.modality_config.responses.sampling_rate = response_sampling_rate

    cfg.dataset.modality_config.screen.sample_stride = 1
    cfg.dataset.modality_config.screen.include_blanks=True
    cfg.dataset.modality_config.screen.valid_condition = {"tier": "train"}
    cfg.dataset.modality_config.screen.transforms.Resize.size = video_size

    cfg.dataloader.num_workers = outer_config['num_workers']
    cfg.dataloader.prefetch_factor = outer_config['prefetch_factor']
    cfg.dataloader.batch_size = batchsize
    cfg.dataloader.pin_memory = True# outer_config['pin_memory']
    cfg.dataloader.shuffle = True
    non_blocking = True # outer_config['non_blocking']

    train_dl = get_multisession_dataloader(full_paths, cfg)


    from experanto.ram_usage import MemoryMonitor
    import psutil
    import time

    # Initialize memory monitor
    memory_monitor = MemoryMonitor()

    n_epochs = 2

    # the first 10 batches are slow because torch is compiling the model for each new input shape
    for epoch in range(n_epochs):
        epoch_fps = []

        for i, (key, batch) in tqdm(enumerate(train_dl), ncols=100, ascii=True):
            # print(f"Batch {i}")
            
            start_time = time.time()
            # print("device = ", device)
            # batch = batch.to(device, non_blocking_pin=True, num_threads=outer_config["num_threads"])
            videos = batch["screen"].to(device, torch.float32, non_blocking=non_blocking).transpose(1,2)
            responses = batch["responses"].to(device, torch.float32, non_blocking=non_blocking)

            # videos = batch["screen"]
            # responses = batch["responses"]

            # End timing
            end_time = time.time()


            # Calculate FPS
            batch_time = end_time - start_time
            num_frames = videos.size(0) * videos.size(2)  # Assuming videos is (batch_size, channels, frames, height, width)
            fps = num_frames / batch_time if batch_time > 0 else float('inf')
            epoch_fps.append(fps)

            if i > 1000:
                break

        # Print summary for the epoch
        avg_fps = sum(epoch_fps) / len(epoch_fps)
        
        # Update the output file after each epoch
        save_results({"avg_fps": avg_fps, 'timestamp_number': time.time(), 'timestamp': datetime.now().isoformat()}, output_file)

        """
        with open(output_file, 'r+') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
            f.seek(0)
            existing_data.append(stats)
            json.dump(existing_data, f, indent=2)
            f.truncate()
        """

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file_prepend', type=str, required=True)
    parser.add_argument('--outer_index', type=int, required=True)
    parser.add_argument('--inner_index', type=int, required=True)
    return parser.parse_args()

sweeps = np.load('sweeps.npy', allow_pickle=True)

if __name__ == "__main__":

    args = parse_args()

    import os
    current_pid = os.getpid()

    outer_index = args.outer_index
    inner_index = args.inner_index

    outer_config = sweeps[outer_index]

    output_file = f"{args.output_file_prepend}/run_{current_pid}.json"


    paths = ['dynamic29513-3-5-Video-full',
            'dynamic29514-2-9-Video-full',
            'dynamic29755-2-8-Video-full',
            'dynamic29647-19-8-Video-full',
            'dynamic29156-11-10-Video-full',
            'dynamic29623-4-9-Video-full',
            'dynamic29515-10-12-Video-full',
            'dynamic29234-6-9-Video-full',
            'dynamic29712-5-9-Video-full',
            'dynamic29228-2-10-Video-full'
            ]

    paths = paths[inner_index*2:(inner_index+1)*2]

    if(inner_index == 0):
        with open(f"{args.output_file_prepend}/outer_config.txt", 'w') as config_file:
            config_file.write(str(outer_config))
                
    train_model(paths, output_file, outer_config, inner_index)