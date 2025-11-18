# -*- coding: utf-8 -*-
"""
https://github.com/LiZhang30/MRBDTA/blob/main/MRBDTA.py
modified with flash attention
"""
import math
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from scipy.signal import max_len_seq


# # Transformer Parameters
# d_model = 128  # Embedding Size
# d_ff = 512  # FeedForward dimension
# d_k = d_v = 32  # dimension of K(=Q), V
# n_layers = 1  # number of Encoder
# n_heads = 4  # number of heads in Multi-Head Attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    # seq_q=seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        d_k = V.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        # attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.fc0 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        n_heads = self.n_heads
        d_k = self.d_k
        d_v = self.d_v
        d_model = self.d_model
        ##residual, batch_size = input_Q, input_Q.size(0)
        batch_size, seq_len, model_len = input_Q.size()
        residual_2D = input_Q.view(batch_size * seq_len, model_len)
        residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.ln(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual)



class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(FlashMultiHeadAttention, self).__init__()
        from flash_attn.modules.mha import MHA
        self.attn = MHA(embed_dim=d_model,
                        num_heads=n_heads, use_flash_attn=True)
        self.ln = nn.LayerNorm(d_model)
        self.fc0 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, enc_inputs, attn_mask):
        from flash_attn.bert_padding import unpad_input, pad_input
        x_unpad, indices, cu_seqlens, max_seqlen, seqused = unpad_input(enc_inputs, (~attn_mask)[:, 0])
        res = self.fc0(enc_inputs)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=enc_inputs.device.type):
            out_unpad = self.attn(x_unpad, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            output = pad_input(out_unpad, indices, enc_inputs.shape[0], enc_inputs.shape[1])
        return self.ln(output+res), None

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        # self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.enc_self_attn = FlashMultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]

        # enc_outputs: [batch_size, src_len, d_model]
        # attn: [batch_size, n_heads, src_len, src_len]
        # enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_k=32, d_v=32, d_ff=512, n_layers=1):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.stream0 = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
        self.stream1 = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
        self.stream2 = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        stream0 = enc_outputs

        enc_self_attns0, enc_self_attns1, enc_self_attns2 = [], [], []
        for layer in self.stream0:
            # enc_outputs: [batch_size, src_len, d_model]
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            stream0, enc_self_attn0 = layer(stream0, enc_self_attn_mask)
            enc_self_attns0.append(enc_self_attn0)

        # skip connect -> stream0
        stream1 = stream0 + enc_outputs
        stream2 = stream0 + enc_outputs
        for layer in self.stream1:
            stream1, enc_self_attn1 = layer(stream1, enc_self_attn_mask)
            enc_self_attns1.append(enc_self_attn1)

        for layer in self.stream2:
            stream2, enc_self_attn2 = layer(stream2, enc_self_attn_mask)
            enc_self_attns2.append(enc_self_attn2)

        return torch.cat((stream1, stream2), 2), enc_self_attns0, enc_self_attns1, enc_self_attns2


class Transformer(nn.Module):
    def __init__(self, drug_vocab_size, target_vocab_size, d_model=128, n_heads=4, d_k=32, d_v=32, d_ff=512, n_layers=1):
        super(Transformer, self).__init__()

        self.encoderD = Encoder(drug_vocab_size, d_model, n_heads, d_k, d_v, d_ff, n_layers)
        self.encoderT = Encoder(target_vocab_size, d_model, n_heads, d_k, d_v, d_ff, n_layers)
        self.fc0 = nn.Sequential(
            nn.Linear(4 * d_model, 16 * d_model, bias=False),
            nn.LayerNorm(16 * d_model),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * d_model, 4 * d_model, bias=False),
            nn.LayerNorm(4 * d_model),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(4 * d_model, 1, bias=False)

    def forward(self, input_Drugs, input_Tars):
        # input: [batch_size, src_len]

        # enc_outputs: [batch_size, src_len, d_model]
        # enc_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_Drugs, enc_attnsD0, enc_attnsD1, enc_attnsD2 = self.encoderD(input_Drugs)
        enc_Tars, enc_attnsT0, enc_attnsT1, enc_attnsT2 = self.encoderT(input_Tars)

        enc_Drugs_2D0 = torch.sum(enc_Drugs, dim=1)
        enc_Drugs_2D1 = enc_Drugs_2D0.squeeze()
        enc_Tars_2D0 = torch.sum(enc_Tars, dim=1)
        enc_Tars_2D1 = enc_Tars_2D0.squeeze()
        fc = torch.cat((enc_Drugs_2D1, enc_Tars_2D1), 1)

        fc0 = self.fc0(fc)
        fc1 = self.fc1(fc0)
        fc2 = self.fc2(fc1)
        affi = fc2.squeeze()

        return affi, enc_attnsD0, enc_attnsT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2
