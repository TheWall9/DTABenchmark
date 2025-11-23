"""modified from https://github.com/Leopold2333/Bi-Mamba4TS/blob/master/layers/BiMamba4TS_layers.py"""
import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2

class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout, residual, drop_flag=1):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.residual = residual
        self.drop_flag = drop_flag

    def forward(self, new, old):
        new = self.dropout(new) if self.drop_flag else new
        return self.norm(old + new) if self.residual else self.norm(new)


class EncoderLayer(nn.Module):
    def __init__(self, mamba_forward, mamba_backward, d_model=128, d_ff=256, dropout=0.2,
                 activation="relu", bi_dir=False, residual=True):
        super(EncoderLayer, self).__init__()
        self.bi_dir = bi_dir
        self.mamba_forward = mamba_forward
        self.residual = residual
        self.addnorm_for = Add_Norm(d_model, dropout, residual, drop_flag=0)

        if self.bi_dir:
            self.mamba_backward = mamba_backward
            self.addnorm_back = Add_Norm(d_model, dropout, residual, drop_flag=0)

        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        )
        self.addnorm_ffn = Add_Norm(d_model, dropout, residual, drop_flag=1)

    def forward(self, x, seq_len=None, seq_idx=None):
        output_forward = self.mamba_forward(x, seq_idx=seq_idx)
        output_forward = self.addnorm_for(output_forward, x)
        if self.bi_dir:
            x_fliped, seq_len_fliped, seq_idx_fliped = self.flip(x, seq_len=seq_len, seq_idx=seq_idx)
            output_backward = self.mamba_backward(x_fliped, seq_idx=seq_idx_fliped)
            output_backward, _, _ = self.flip(output_backward, seq_len=seq_len_fliped, seq_idx=seq_idx_fliped)
            output_backward = self.addnorm_back(output_backward, x)
            output = output_forward + output_backward
        else:
            output = output_forward
        temp = output
        output = self.ffn(output.transpose(-1, 1)).transpose(-1, 1)
        output = self.addnorm_ffn(output, temp)
        return output

    def flip(self, x, seq_len=None, seq_idx=None):
        if seq_idx is not None:
            return torch.flip(x, dims=[1]), None, torch.flip(seq_idx, dims=[1])
        if seq_len is None:
            return torch.flip(x, dims=[1]), None, None
        else:
            new_x = torch.zeros_like(x)
            for i, (row, length) in enumerate(zip(x, seq_len)):
                new_x[i, :length] = torch.flip(row[:length], dims=[0])
            return new_x, seq_len, None


class MambaEncoder(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 n_layer: int = 1,
                 d_ff: int = 256,
                 dropout: float = 0.2,
                 activation: str = "relu",
                 residual: bool = True,
                 bidirectional: bool = True,
                 ssm_cfg=None):
        super(MambaEncoder, self).__init__()
        if ssm_cfg is None:
            ssm_cfg = {}
        ssm_cfg['layer'] = ssm_cfg.get('layer', 'Mamba1')
        if ssm_cfg['layer'] == "Mamba2":
            headdim = ssm_cfg.get('headdim', 64)
            expand = ssm_cfg.get('expand', 2)
            if (expand*d_model)%(headdim*8)!=0:
                assert (expand*d_model)%8==0
                ssm_cfg['headdim'] = (expand*d_model)//8
        self.d_model = d_model
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.ssm_cfg = ssm_cfg.copy()
        self.layers = nn.ModuleList()
        mamba_cls = Mamba2 if ssm_cfg.pop('layer') == 'Mamba2' else Mamba
        for i in range(n_layer):
            mamba_forward = mamba_cls(d_model=d_model, **ssm_cfg)
            mamba_backward = mamba_cls(d_model=d_model, **ssm_cfg) if bidirectional else None
            self.layers.append(EncoderLayer(mamba_forward, mamba_backward, d_model,
                                            d_ff=d_ff, activation=activation, residual=residual,
                                            bi_dir=bidirectional, dropout=dropout))
    def forward(self, x, seq_len=None, seq_idx=None):
        for layer in self.layers:
            x = layer(x, seq_len=seq_len, seq_idx=seq_idx)
        return x

    def binding_ssm(self, other):
        for ref, layer in zip(self.layers, other.layers):
            ref.mamba_forward.A_log = layer.mamba_forward.A_log
            if getattr(ref, "mamba_backward", None) is not None:
                ref.mamba_backward.A_log = layer.mamba_backward.A_log

class Encoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer

    def forward(self, x):
        # [B, S, D]
        for mamba_block in self.mamba_layers:
            x = mamba_block(x)

        if self.norm is not None:
            x = self.norm(x)

        return x