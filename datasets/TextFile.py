import os
import json
import random
import numpy as np
import torch
from utils.registry import Dataset
from utils.registry import CommandLineArgument, TrainingArguments
from dataclasses import dataclass

TrainingArguments.append(CommandLineArgument(str, 'srcpath', 'Path to the dataset to be loaded', None, True))
TrainingArguments.append(CommandLineArgument(float, 'train_split', 'Percentage of dataset to use for training', 0.9))

class TextFile(metaclass=Dataset):
    def __init__(self, config, split:str='train'):
        data = np.memmap(config.srcpath, mode='r')
        idx = int(len(data)*config.train_split)
        self.data = data[:idx] if split=='train' else data[idx:]
        self.config = config
        
    def to(self, device:str):
        self.device = device
    
    def random(self):
        seq_len = self.config.max_length
        ix = torch.randint(len(self.data) - seq_len - 1, (self.config.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
        if self.device == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
    
class TextMultiSet(metaclass=Dataset):
    def __init__(self, config, split:str='train'):
        self.config = config
        index_path = os.path.join(config.srcpath, 'index.json')
        if not os.path.exists(index_path):
            TextMultiSet.prepare_index(config.srcpath, config.train_split)
        with open(index_path) as f:
            self.index = json.load(f)[split]

    def to(self, device:str):
        self.device = device

    def random(self):
        docidx = random.randint(0, len(self.index) - 1)
        docpath = os.path.join(self.config.srcpath, self.index[docidx])
        data = np.memmap(docpath, mode='r')

        seq_len = self.config.max_length
        l = min(len(data)-1, seq_len)
        ix = torch.randint(len(data) - l - 1, (self.config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+l]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+l]).astype(np.int64)) for i in ix])
        if self.device == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
    
    @dataclass
    class iterator_state:
        docidx: int = 0
        cursor: int = 0
        data: np.ndarray = None

    def __iter__(self):
        self.state = [TextMultiSet.iterator_state() for _ in range(self.config.batch_size)]
        return self
    
    def __next__(self):
        for s in self.state:
            if s.data is None or s.cursor >= len(s.data):
                self.__nextdoc(s)

        seq_len = self.config.max_length
        for s in self.state:
            doclen = len(s.data)
            seq_len = min(doclen-1, seq_len)
            s.cursor = min(doclen-seq_len-1, s.cursor)

        x = torch.stack([torch.from_numpy((s.data[s.cursor:s.cursor+seq_len]).astype(np.int64)) for s in self.state])
        y = torch.stack([torch.from_numpy((s.data[s.cursor+1:s.cursor+1+seq_len]).astype(np.int64)) for s in self.state])

        if self.device == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        
        for s in self.state:
            s.cursor = s.cursor + random.randint(1, 2*seq_len)
            if s.cursor >= (len(s.data) - self.config.max_length):
                s.cursor = len(s.data)

        return x, y

    def __nextdoc(self, state):
        docidx = random.randint(0, len(self.index) - 1)
        docpath = os.path.join(self.config.srcpath, self.index[docidx])
        data = np.memmap(docpath, mode='r')
        
        print(f'Reading {self.index[docidx]}')
        
        state.docidx = docidx
        state.data = data
        state.cursor = 0

    @classmethod
    def prepare_index(cls, srcpath, train_split):
        print(f'Preparing index at {srcpath}')
        index_path = os.path.join(srcpath, 'index.json')
        files = [f for f in os.listdir(srcpath) if os.path.isfile(os.path.join(srcpath, f))]
        random.shuffle(files)
        split = int(len(files)*train_split)
        print(f'Found {len(files)} files - randomly assigning {split} files to the training set and the rest to the validation set')
        dataset = {
            'train': files[:split],
            'validation': files[split:]
        }
        with open(index_path, 'w') as f:
            json.dump(dataset, f)
        return