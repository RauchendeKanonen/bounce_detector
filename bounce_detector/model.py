from __future__ import annotations
import torch
import torch.nn as nn

class TinyTemporalCNN(nn.Module):
    def __init__(self, F: int, T: int, hidden: int = 64, n_out: int = 1, k: int = 9, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.F, self.T = F, T
        self.hidden = hidden
        self.k = k
        self.layers = layers
        self.dropout = dropout
        width = hidden
        mods = []
        in_ch = F
        for _ in range(layers):
            mods += [
                nn.Conv1d(in_ch, in_ch, k, padding=k//2, groups=in_ch),
                nn.Conv1d(in_ch, width, 1),
                nn.GELU(),
                nn.BatchNorm1d(width),
                nn.Dropout(dropout),
            ]
            in_ch = width
        self.tcn = nn.Sequential(*mods)
        self.head = nn.Sequential(
            nn.Linear(width, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_out)
        )

    def forward(self, x):
        B,T,S,F = x.shape
        x = x.view(B, T, F)
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h_last = h[:, :, -1]
        return self.head(h_last)
