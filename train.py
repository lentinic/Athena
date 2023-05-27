import os
import time
import argparse
import inspect
import json
import torch
import math
from pathlib import Path

from torch.nn import functional as F

from dataclasses import dataclass
from utils.registry import Model, Dataset, TrainingArguments
from models import GPT2, ButterflyGPT
from datasets import Shakespeare, TextFile
from contextlib import nullcontext

@dataclass
class TrainingConfig:
    log_interval: int = 10
    validation_interval: int = 1000
    validation_iters: int = 100
    batch_size: int = 64
    max_length: int = 256
    gradient_accumulation_steps: int = 1
    model: str = 'GPT2'
    name: str = 'athena'
    dataset: str = None
    srcpath: str = '../datasets/tinyshakespeare.txt'
    learning_rate: float = 5e-4
    rate_decay: float = 0.99
    warmup_iters: int = 2000
    decay_iters: int = 600000                   # should be ~= max_iters per Chinchilla
    min_rate: float = 5e-5                      # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
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
    optimizer = checkpoint['optimizer']
    config = checkpoint['config']
    state = checkpoint['state']

    model = CreateModel(config.model)
    print(checkpoint['model'])
    model.load_state_dict(checkpoint['model'])

    return model, optimizer, config, state

def SaveCheckpoint(outdir, model, optimizer, config, state):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'state': state
    }

    outdir = os.path.join(outdir, config.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(f'Saving checkpoint to {outdir}')
    
    torch.save(checkpoint, os.path.join(outdir, f'iteration-{state.iter}.ckpt'))

def CreateModel(modelname):
    if modelname not in Model.registry:
        raise Exception(f'Unknown model "{modelname}"')
    return Model.registry[modelname]()

def ConfigureOptimizer(model, config):
    fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    print(f'Configuring AdamW optimizer - fused = {fused}')

    nodecay = { 'params': [], 'weight_decay': 0.0 }
    optim_groups = []

    for mn, m in model.named_modules():
        if len(mn) == 0:
            continue
        for pn, p in m.named_parameters():
            if not hasattr(p, '_optim'):
                nodecay['params'].append(p)
                continue
            
            optim_groups.append(
                { 'params': [p], 'weight_decay': p._optim['wd'] }
            )
    
    optim_groups.append(nodecay)
    extra_args = dict(fused=True) if fused else dict()
    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2), **extra_args)

def PrepareDataset(config):
    if config.dataset not in Dataset.registry:
        raise Exception(f'Unkown dataset "{config.dataset}"')
    return Dataset.registry[config.dataset](config)

def RunValidationIteration(device, ctx, model, dataset, config, state):
    model.eval()
    losses = torch.zeros(config.validation_iters)
    for k in range(config.validation_iters):
        x, y = dataset.random()
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def UpdateLearningRate(optimizer, state, config):
    if state.iter < config.warmup_iters:
        lr = config.min_rate + ((config.learning_rate - config.min_rate) * state.iter / config.warmup_iters)
    elif state.iter > config.decay_iters:
        lr = config.min_rate
    else:
        ratio = (state.iter - config.warmup_iters) / (config.decay_iters - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        lr = config.min_rate + coeff * (config.learning_rate - config.min_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a language model')
    parser.add_argument('--resume', type=argparse.FileType('rb'), help='Path to a checkpoint to resume training from')
    parser.add_argument('--dataset', type=str, help='Name of the dataset to train on')
    parser.add_argument('--model', type=str, help='Name of the model class to train')
    parser.add_argument('--seed', type=int, default=1337, help="Random seed to init torch with")
    parser.add_argument('--device', type=str, default='cuda', help='Device to train model on')
    parser.add_argument('--compile', type=bool, default=True, help='Compile model before training or not')
    parser.add_argument('--name', type=str, default='Athena', help='Base output name for saving checkpoints')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--outdir', type=str, default='checkpoints')

    for arg in TrainingArguments:
        parser.add_argument(f'--{arg.name}', type=arg.value_type, default=arg.default, required=arg.required, choices=arg.choices, help=arg.help)

    arguments = parser.parse_args()

    try:
        if arguments.model is None and arguments.resume is None: raise Exception('No model name specified for training')
        if arguments.dataset is None and arguments.resume is None: raise Exception('No dataset name specified for training')
    except Exception as e:
        print(f'Command line error: {str(e)}')
        parser.print_help()
        quit()

    if arguments.resume is not None:
        model, optimizer_state, config, state = LoadCheckpoint(arguments.resume)
        if arguments.dataset is not None:
            config.dataset = arguments.dataset
        config.seed = arguments.seed
    else:
        model = CreateModel(arguments.model)
        optimizer_state = None
        config = TrainingConfig(model = arguments.model, name = arguments.name, dataset = arguments.dataset, seed = arguments.seed)
        state = TrainingState()
    
    if arguments.srcpath is not None:
        config.srcpath = arguments.srcpath

    n_params = sum(p.numel() for p in model.parameters())
    n_params -= model.emb.weight.numel()
    print(f'Training model with {n_params/1e6:.2f} M parameters')

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

    optimizer = ConfigureOptimizer(model, config)
    if not optimizer_state is None:
        optimizer.load_state_dict(optimizer_state)
        
    if arguments.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f'Model compilation failed - {str(e)}')
            pass
    
    dataset = PrepareDataset(config)
    dataset.to(device)

    t0 = time.time()
    train_iter = iter(dataset)
    x, y = next(train_iter)
    while True:
        state.iter = state.iter + 1

        UpdateLearningRate(optimizer, state, config)

        for step in range(config.gradient_accumulation_steps):
            with ctx:
                z, training_loss = model(x, y)
            x, y = next(train_iter)
            scaler.scale(training_loss).backward()
        
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if state.iter % config.validation_interval == 0:
            validation_loss = RunValidationIteration(device, ctx, model, dataset, config, state)
            if (validation_loss < state.validation_loss):
                state.validation_loss = validation_loss
                SaveCheckpoint(arguments.outdir, model, optimizer, config, state)

        if state.iter % config.log_interval == 0: 
            t1 = time.time()
            delta = (t1 - t0) * 1000
            t0 = t1

        if state.iter % config.validation_interval == 0:
            print(f'iter {state.iter}: time {delta:.2f}ms, loss {training_loss:.4f}, validation {validation_loss:4f}')
        elif state.iter % config.log_interval == 0:
            print(f'iter {state.iter}: time {delta:.2f}ms, loss {training_loss:.4f}')
        
        if state.iter >= config.max_iters:
            break