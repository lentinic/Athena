import torch
from torch import nn
import torch.nn.functional as F
from utils.registry import Model
from utils import optimization_config
import math
from torch_butterfly import Butterfly

# ripped from Andrej Karpathy's nanogpt with some modification
# https://github.com/karpathy/nanoGPT
# nanogpt is available under MIT license

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim:int, heads:int, seqlen:int = 256, bias:bool = False, dropout_p:float = 0.2):
        super().__init__()
        
        assert dim % heads == 0

        self.heads = heads
        self.dim = dim
        self.dropout_p = dropout_p
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Butterfly(dim, 3 * dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout_p)
        self.resid_dropout = nn.Dropout(dropout_p)
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(seqlen, seqlen)).view(1, 1, seqlen, seqlen))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.dim, dim=2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            z = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            z = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        z = z.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        z = self.resid_dropout(self.c_proj(z))
        return z

class MLP(nn.Module):
    def __init__(self, dim:int, bias:bool = True, dropout_p:float = 0.2):
        super().__init__()
        self.c_fc    = nn.Linear(dim, 4 * dim, bias=bias)
        self.c_proj  = Butterfly(4 * dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, dim:int, heads:int, seqlen:int = 256, bias:bool = False, dropout_p:float = 0.2):
        super().__init__()
        self.ln_1 = LayerNorm(dim, bias=bias)
        self.attn = CausalSelfAttention(dim, heads, seqlen, bias, dropout_p)
        self.ln_2 = LayerNorm(dim, bias=bias)
        self.mlp = MLP(dim, bias=bias, dropout_p=dropout_p)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTBase(nn.Module):
    def __init__(self, layers:int, dim:int, heads:int, seqlen:int, bias:bool, dropout_p:float):
        super().__init__()
        
        self.emb = nn.Embedding.from_pretrained((torch.rand(256, dim) - 0.5).sign())
        self.transformer = nn.Sequential(
            nn.Dropout(dropout_p)
        )

        for _ in range(layers):
            self.transformer.append(Block(dim, heads, seqlen, bias, dropout_p))
        
        self.transformer.append(LayerNorm(dim, bias=bias))
        self.transformer.append(nn.Linear(dim, 256, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        z = self.transformer(self.emb(x))
        if y is not None:
            return z, F.cross_entropy(z.view(-1, z.size(-1)), y.view(-1), ignore_index=-1)
        return z, None
    
class ButterflyGPT(GPTBase, metaclass=Model):
    def __init__(self):
        super().__init__(6, 384, 6, 256, False, 0.2)