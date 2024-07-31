import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
# from hellaswag import render_example, iterate_examples

@dataclass 
class GPTConfig: #basically an enum 
    block_size: int = 1024 #the maximum sequence length inputted into GPT
    vocab_size: int =  50257#the max number of words 
    n_layer: int = 12 #number of decoder layers (casual, sel attention, mlp and the pre-add norms for each respective)
    n_heads: int = 12
    n_embed = 768 


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0 
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3) #this is named conventionally similar to hugging face import. All of these layers have weights except the buffer. 
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_heads
        self.d_head =  config.n_embed // config.n_heads 
        self.n_embed = config.n_embed 
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size)) 
        #buffers are not a part of hte models parameters so it's not updated during training. View here is just unsqueeze(0)

    def forward(self, x):
        B, T, C = x.shape #batch, seq_len, (block_size),  embedding size (n_embed) 
        Q, K, V = torch.chunk(self.c_attn(x), 3, dim=-1) #OR as indicated in video self.c_attn(x).split(self.n_embed, dim=2) #split size and the dim to split on? 
        Q = Q.view(B, T , self.n_heads, self.d_head).transpose(1,2) #making the n_head dim like batch so computed in || 
        K = K.view(B, T , self.n_heads, self.d_head).transpose(1,2) 
        V = V.view(B, T , self.n_heads, self.d_head).transpose(1,2) 
        attention = (Q @ K.transpose(-1,-2)) / (math.sqrt(self.d_head))
        attention = attention.masked_fill(self.bias[:, :, T, T] == 0, float('-inf')) #is this not the entire tensor in any case??? Unless T is not actually config.block_size, but rather the size of text. 
        attention = F.softmax(attention, dim=-1)
        y = attention @ V #this is a weighted sum basically of the attention scores.
        y = y.transpose(1,2).contiguos().view(B,T,C) #concatenate all of it back
        return self.c_proj(y) 



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate='tanh') #approximate not necessary any more. GELU like RELU but no dead weights i.e gradient != 0
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

#Very similar to Vaswani 2017 transformer but...
# 1. addnorms are prior to the main layer not after. 
# 2. Residuals are added direct without being put in norm so there's a path in the gradients to the raw input.    
#TODO: Where is the cross attention? --> DOESN'T EXIST! There's no encoder! 
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.ln_2 = config.n_embed
        self.attention = CasualSelfAttention(config) #i.e masked self attention - the naming convention for the params therefore becomes or ends with attention.bias for the bias buffer. 
        self.mlp = MLP() #aka feed forward nn 
    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 



class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), #the word embeddings that are learnt
            wpe = nn.Embedding(config.block_size, config.n_embed), #the positional mebeddings that are added, basically the prev sine and cosine waves that are learnt now. Basically one embedding per index
            h = nn.ModuleList([Block(config) for i in config.n_layer]), #this is the transformer blocks 
            ln_f = nn.LayerNorm(config.n_embed) #this is just normalisation layer. 
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) #this is the classification layer.
    
    def forward(self, idx, targets=None):
        B, T = idx.size() #this is the input of shape B, T -> can also do .shape )i's post tokenisation. 
        assert T <= self.config.block_size, f"Cannot handle something more than the block size limit - max sq length"
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)

        #1D positions (T) -> (T, n_embed)
        pos_emb = self.transformer.wpe(positions) 

        #(B, T) -> (B, T, n_embed)
        word_embeddings = self.transformer.wte(idx) 

        x = pos_emb + word_embeddings #broadcast, (1, T, n_embed) to (B, T, n_embed)

        for block in self.transformer.h:
           x = block(x)
        
        x = self.transformer.ln_f(x)

        # (B, T, vocab_size)
        x = self.lm_head(x)

        loss = F.cross_entropy(x.view(-1, self.config.vocab_size), targets.view(-1)) #this is just the way the cross entropy function works. No need to softmax yet. It will expect (B*T, vocab) and (B*T) https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        return x, loss
    
    @classmethod 
    def from_pretrained(cls, model_type, override_args = None ):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt-xl'}
        override_args = override_args or {} 
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024 
        config = GPTConfig(**config_args)
        model = GPT(config) #weird, doesn't take in any params 
        sd = model.state_dict() 
        sd_keys = sd.keys()
        sd_keys = [key for key in sd_keys if not key.endswith('.attention.bias')]

        #init a hugging face/transformers model 

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
         # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for key in sd_keys_hf:
            if any([key.endswith(k) for k in transposed]):
                assert sd_hf[key].shape[::-1] == sd[key].shape #i.e ensuring reversed shape is the same. Fully reversed not just transpose 1,2 
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t()) #we are copying the weights sd[key] is a tensor
            else:
                 assert sd_keys_hf.shape == sd_keys[key].shape
                 sd[key].copy_(sd_keys_hf[key])
        
        return model
    

# -------------------------------------------------------------------------------------------------------
num_return_sequences = 5 
max_length = 30

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends('mps')) and torch.backends.mps.is_available():
    device = 'mps'
print('device', device)

model = GPT.from_pretrained('gpt2')
#to train from scratch we comment out prior line and do model = GPT(GPTConfig()) #params are randomly initialised by default with pytorch ie Xavieri nitialisation. 
model.eval()
model.to(device)
print("didn't crash yay!")

import tiktoken 
enc = tiktoken.get_encoding('gpt2')
#---- data loader -----#
with open('input.txt', 'r') as file:
    data = file.read()
data = data[:1000]
data = enc.encode(data)
B, T = 4, 32
buf = torch.tensor(data[:B*T + 1]) #first we get the necessary amount of tokens  (i.e get 1000 sample, encode, get subset of B*T + 1, then split into x,y tensors then initialise model and put in)
buf = buf.to(device)
x = buf[:-1].view(B,T) 
y = buf[1:].view(B,T)

#get logits
model = GPT(GPTConfig())
model.to(device)
# y_pred, loss = model(x, y) 
# print(loss) #we expect the loss here to be around -log(1/vocab_size) since in the beginning should be randomly initialised and equally probable. i.e cross entropy loss 
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    y_pred, loss = model(x,y)
    optimizer.step()
    print(loss.item())


import sys ; sys.exit() 



tokens = enc.encode('Hello, I am a language model, ')
tokens = torch.tensor(tokens, dtype=torch.long) #(8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #params are the repeats per dimension!  -> 5,8 or B,8 
x = tokens.to(device)

#generation s
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#only care about last token prediction 
while len(x) < max_length:
    output = model(x) #(B,T,Vocab_size)
    last_output = output[:, -1, :].squeeze() #(B, vocab_size)
    probs = F.softmax(last_output, dim=-1)

    topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1) #(5,50) or B,50 topk_indicies: The indices of these top 50 probabilities in the original vocabulary.
    #select token from top-k probs 
    chosen_token_ix = torch.multinomial(topk_probs, 1) #(B, 1) 1 = 1 sample to draw Just plain sampling based on probs 
    #gather corresponding indices
    xcol = torch.gather(topk_indicies, -1, chosen_token_ix) #(B,1) -> I think this is getting the indices in the original probs vector by looking at the chosen indexes in topk probs. 
    #takes in input, dim, index. Indices = the token number. 
    x = torch.cat([x, xcol], dim=1) #I think -1 here works also since x is B,T


    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist() #selecting all but within certain batch + tensor to list
        decoded = enc.decode(tokens)
        print('>', decoded)


    #do top-k sampling i.e only look at the 50 highest 









    
    