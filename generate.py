import os
import time
import argparse
import inspect
import json
import torch
from pathlib import Path

from torch.nn import functional as F

from dataclasses import dataclass
from utils.registry import Model, Dataset
from models import GPT2, ButterflyGPT
from contextlib import nullcontext

@dataclass
class TrainingConfig:
    log_interval: int = 10
    validation_interval: int = 200
    validation_iters: int = 100
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    model: str = 'GPT2'
    name: str = 'athena'
    dataset: str = None
    learning_rate: float = 6e-4
    rate_decay: float = 0.99
    decay_iters: int = 600000                   # should be ~= max_iters per Chinchilla
    min_rate: float = 6e-5                      # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    weight_decay: float = 1e-1                  
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
    max_iters: int = 5000

@dataclass
class TrainingState:
    iter: int = 0
    validation_loss: float = 1e10

def LoadCheckpoint(file, device='cuda'):
    checkpoint = torch.load(file, map_location=device)
    config = checkpoint['config']
    model = CreateModel(config.model)
    model.load_state_dict(checkpoint['model'])

    return model, config

def CreateModel(modelname):
    if modelname not in Model.registry:
        raise Exception(f'Unknown model "{modelname}"')
    return Model.registry[modelname]()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load language model and generate text')
    parser.add_argument('--checkpoint', type=argparse.FileType('rb'), help='Path to a checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=1337, help="Random seed to init torch with")
    parser.add_argument('--device', type=str, default='cuda', help='Device to train model on')
    parser.add_argument('--compile', type=bool, default=True, help='Compile model before training or not')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--prompt', type=str, default='\n')
    parser.add_argument('--length', default=128, type=int)
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--temperature', default=0.8, type=float)

    arguments = parser.parse_args()

    model, config = LoadCheckpoint(arguments.checkpoint)
    config.seed = arguments.seed
    
    n_params = sum(p.numel() for p in model.parameters())
    n_params -= model.emb.weight.numel()
    print(f'Model with {n_params/1e6:.2f} M parameters')

    device = arguments.device
    dtype = arguments.dtype
    device_type = 'cuda' if 'cuda' in device else 'cpu' 

    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True    # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True          # allow tf32 on cudnn

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model.to(device)
        
    if arguments.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f'Model compilation failed - {str(e)}')
            pass

    x = torch.ByteTensor(list(bytes(arguments.prompt, 'utf8'))).type(torch.int64).to(device)
    x = x.unsqueeze(0)
    decode = lambda l: ''.join([chr(i) for i in l.tolist()[0]])

    with torch.no_grad():
        with ctx:
            for _ in range(arguments.num_samples):
                sample = x
                for _ in range(arguments.length):
                    logits, _ = model(sample)
                    logits = logits[:, -1, :] / arguments.temperature
                    probs = F.softmax(logits, dim=-1)
                    s = torch.multinomial(probs, num_samples=1)
                    sample = torch.cat((sample, s), dim=1)
                print('------')
                print(decode(sample))

