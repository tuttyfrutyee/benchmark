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


class MouseHiera(nn.Module):
    def __init__(self,
                backbone,
                dls,
                chunk_size,
                dim=192,
                dim_head=32,
                num_heads=4,
                mlp_ratio=4,):
        super().__init__()
        self.backbone=backbone
        self.num_heads=num_heads
        self.dim_head=dim_head
        self.dim=dim
        self.dim_q = dim_head*num_heads
        self.wq = nn.Linear(self.dim_q, self.dim_q, bias=False)
        self.wk = nn.Linear(dim, self.dim_q, bias=False)
        self.wv = nn.Linear(dim, self.dim_q, bias=False)
        self.wo = nn.Linear(self.dim_q, self.dim_q, bias=False)
        
        self.neuron_proj = nn.Linear(self.dim_q, chunk_size, bias=False)
        
        
        self.kv_norm=torch.nn.RMSNorm(dim)
        self.q_norm=torch.nn.RMSNorm(self.dim_q)
        self.qkv_norm=torch.nn.RMSNorm(self.dim_q)
        self.mlp = MLP(dim=self.dim_q, hidden_dim=int(self.dim_q * mlp_ratio))
        self.readout = nn.ModuleDict()
        self.activation = nn.Softplus(beta=0.1) # probably a much better activation than ELU+1
        for k, v in dls.loaders.items():
            n_neurons = next(iter(v))["responses"].shape[-1]
            self.readout[k] = IndexedLinearReadout(n_neurons, 
                                                in_features=dim_head*num_heads,
                                                dim_head=dim_head, 
                                                num_heads=num_heads, 
                                                )
        self.init_weights()

    def forward(self, x, key):
        x = self.backbone(x, return_intermediates=True)[-1][-1]
        b, t, h, w, d = x.shape
        x = self.kv_norm(x)
        x = x.view(b, -1, d) # (B, t*h*w, D)
        k, v = self.wk(x), self.wv(x)
        q = self.q_norm(self.readout[key].query) # (1, N, D)
        q = q.repeat(b, 1, 1) # repeat query for number of batches
        q_attn = self.wq(q)
        q_attn = q_attn.view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.dim_head).transpose(1, 2) # (B, H, S, D)
        v = v.view(b, -1, self.num_heads, self.dim_head).transpose(1, 2) # (B, H, S, D)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            o = F.scaled_dot_product_attention(q_attn, k, v)
        # (B, H, S, D) -> (B, N, D), with N = num_neurons
        o = o.transpose(1,2).contiguous().view(b, -1, self.dim_q)
        o = self.wo(o) + q
        o = self.qkv_norm(o)  
        o = self.mlp(o) + o
        o = self.neuron_proj(o) # (B, N, D) -> (B, N, t)
        o = o + self.readout[key].bias
        o = self.activation(o)
        return o
    
    def init_weights(self, std=.5, cutoff_factor: int = 3):
        """See `TorchTitan <https://github.com/pytorch/torchtitan/blob/40a10263c5b3468ffa53b3ac98d80c9267d68155/torchtitan/models/llama/model.py#L403>`__."""
        std = self.dim_q**-0.5
        for lin in (self.wq, self.wk, self.wv, self.wo):
            nn.init.trunc_normal_(
                lin.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )


# In[9]:


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.init_weights()

    def forward(self, x):
        return self.net(x)
        
    def init_weights(self, std=.5, cutoff_factor: int = 3):
        """See `TorchTitan <https://github.com/pytorch/torchtitan/blob/40a10263c5b3468ffa53b3ac98d80c9267d68155/torchtitan/models/llama/model.py#L403>`__."""
        nn.init.trunc_normal_(
            self.net[0].weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
        nn.init.trunc_normal_(
            self.net[2].weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
        self.net[0].bias.data.zero_()
        self.net[2].bias.data.zero_()
        


# In[10]:


class IndexedLinearReadout(nn.Module):
    """
    Readout module for MTM models with selectable weights based on 
    input IDs. Based on :class:`torch.nn.Linear`.
    """
    def __init__(
        self,
        unique_ids: int,
        in_features: int = 384,
        dim_head=32,
        num_heads=4,
        bias: bool = True,
        device="cuda",
        dtype=torch.float32,
        init_std: float = 0.02,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.unique_ids = unique_ids
        self.in_features = in_features
        self.init_std = init_std
        self.query = nn.Parameter(
            torch.empty(1, unique_ids, dim_head*num_heads, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(1, unique_ids, 1, **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self, cutoff_factor: int = 3):
        """See `TorchTitan <https://github.com/pytorch/torchtitan/blob/40a10263c5b3468ffa53b3ac98d80c9267d68155/torchtitan/models/llama/model.py#L403>`__."""
        readout_std = self.in_features**-0.5
        nn.init.trunc_normal_(
            self.query,
            mean=0.0,
            std=readout_std,
            a=-cutoff_factor * readout_std,
            b=cutoff_factor * readout_std,
        )
        if self.bias is not None:
            self.bias.data.zero_()





# In[2]:


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


    # ### get Hiera backbone

    # In[7]:


    # pip install hiera-transformer
    from hiera import Hiera
    tiny_hiera = Hiera(input_size=(screen_chunk_size, video_size[0], video_size[1]),
                        num_heads=1,
                        embed_dim=96,
                        stages=(2, 1,), # 3 transformer layers 
                        q_pool=1, 
                        in_chans=1,
                        q_stride=(1, 1, 1,),
                        mask_unit_size=(1, 8, 8),
                        patch_kernel=(5, 5, 5),
                        patch_stride=(3, 2, 2),
                        patch_padding=(1, 2, 2),
                        sep_pos_embed=True,
                        drop_path_rate=drop_path_rate,
                        mlp_ratio=4,)

    tiny_hiera = tiny_hiera.to(device, torch.float32);
    example_input = torch.ones(8,1,screen_chunk_size, 36,64).to(device, torch.float32)
    out = tiny_hiera(example_input, return_intermediates=True);

    hiera_output = out[-1][-1]
    hiera_output.shape # (b, t, h, w, c): (8, 4, 9, 16, 192)


    # # Model definition

    # In[8]:


    # ### Build Model

    # In[11]:


    backbone_dim = hiera_output[-1][-1].shape[-1]
    model = MouseHiera(backbone=tiny_hiera, 
                            dls=train_dl, 
                            chunk_size=response_chunk_size,
                            dim=backbone_dim, 
                            dim_head=dim_head,
                            num_heads=num_heads,
                        mlp_ratio=mlp_ratio)
    model = model.to(torch.float32).to(device);


    # # Trainer

    # In[12]:


    # pip install -U pytorch_warmup
    import pytorch_warmup as warmup

    n_epochs = 2
    lr = 2e-4
    gradient_clipping = 1.0
    criteria = nn.PoissonNLLLoss(log_input=False, reduction='mean')
    opt = AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                            T_max=1e6, 
                                                            eta_min=1e-5)
    warmup_scheduler = warmup.UntunedLinearWarmup(opt)


    # # train

    # In[13]:


    from experanto.ram_usage import MemoryMonitor
    import psutil
    import time

    # Initialize memory monitor
    memory_monitor = MemoryMonitor()

    # the first 10 batches are slow because torch is compiling the model for each new input shape
    for epoch in range(n_epochs):
        epoch_fps = []

        for i, (key, batch) in tqdm(enumerate(train_dl), ncols=100, ascii=True):
            # print(f"Batch {i}")
            
            start_time = time.time()

            videos = batch["screen"].to(device, torch.float32, non_blocking=non_blocking).transpose(1,2)
            responses = batch["responses"].to(device, torch.float32, non_blocking=non_blocking)
            
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(videos, key);
            loss = criteria(out.transpose(1,2), responses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping, norm_type=2)
            opt.step()
            opt.zero_grad()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

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