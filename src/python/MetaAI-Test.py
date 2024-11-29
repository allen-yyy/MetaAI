import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import struct
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import sys

src_vocab_size = 25002
d_model = 512
n_layers = 6
n_head = 8
d_k = 64
d_v = 64
d_ff = 2048
tgt_vocab_size = 25002
tgt_len = 25002
n_context = 8
d_context = 64

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        #out = self.gamma * out + self.beta
        return out

def get_attn_pad_mask(seq_q, seq_k): 
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(25001).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(src_vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)

def get_sinusoid_encoding_table(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale_factor, dropout=0.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.scale_factor, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 1, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(scale_factor=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, 
                                               enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.src_emb = Embeddings(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs).cuda()
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decider(nn.Module):
    def __init__(self):
        super(Decider, self).__init__()
        
        self.dec_context_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.dec_input_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.context = torch.Tensor(n_context * d_context, d_model)
        self.contexts = torch.stack([self.context for _ in range(1)]).cuda()
    
    def clean(self, dim = 16):
        self.contexts = torch.stack([self.context for _ in range(dim)]).cuda()

    def forward(self, dec_inputs, enc_inputs):
        #self.contexts.requires_grad_(False)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        dec_outputs, attn = self.dec_input_attn(dec_inputs, self.contexts, self.contexts)
        contexts, attn = self.dec_context_attn(self.contexts, dec_inputs, dec_inputs, enc_self_attn_mask)
        return dec_outputs

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, 
                                                 dec_inputs)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, 
                                                enc_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = Embeddings(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs).cuda()

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class MetaAI(nn.Module):
    def __init__(self):
        super(MetaAI, self).__init__()
        self.encoder = Encoder()
        self.decider = Decider()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs):
        #self.decider.clean(enc_inputs1.size(0))
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        #enc_outputs = self.decider(enc_outputs)
        dec_outputs = torch.tensor([[25000]]).cuda()
        head = torch.tensor([[25000]]).cuda()
        numb = 0
        while(True):
            dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_outputs, enc_inputs, enc_outputs)
            dec_logits = self.projection(dec_outputs)
            dec_logits = dec_logits.view(-1,dec_logits.size(-1))
            #print(dec_logits.shape)
            #dec_logits = F.softmax(dec_logits,dim=1)
            dec_outputs = torch.argmax(dec_logits, dim=1)
            print(dec_outputs[dec_outputs.size(0)-1])
            if dec_outputs[dec_outputs.size(0)-1]==0:
                break
            numb+=1
            if numb > 50:
                break
            dec_outputs = dec_outputs.unsqueeze(0)
            dec_outputs = torch.cat([head, dec_outputs],dim=1)
            #print(dec_outputs.shape)
            
        return dec_outputs

AI=MetaAI()
AI.load_state_dict(torch.load('MetaAI4.pth',weights_only = True),strict=False)
AI.to("cuda")
AI.eval()

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my-new-tokenizer")
tokenizer.add_tokens(['<|beginoftext|>','<pad>'])
tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>'})
tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

while(True):
    print('>',end='')
    line = sys.stdin.readline().strip()
    line = [line]
    data_input = tokenizer(line,padding='longest')['input_ids']
    data_input = torch.Tensor(data_input).long()
    dec_outputs = AI(data_input.cuda())
    print('[',end='')
    print(','.join(str(item) for item in dec_outputs.tolist()),end='')
    print(']')
    answer = tokenizer.convert_ids_to_tokens(dec_outputs.tolist())
    print(str(answer))
