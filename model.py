import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)


class CNN_ChaosClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=3, stride=1, padding=2, num_classes=2):
        super(CNN_ChaosClassifier, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.act_layer = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.act_layer(self.bn1(self.conv1(x)))
        x = self.act_layer(self.bn2(self.conv2(x)))
        x = self.global_avg_pool(x).squeeze(-1)

        x = self.fc1(x)
        x = self.dropout(self.act_layer(x))
        x = self.fc2(x)
        return x


class LSTM_ChaosClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1,
                 num_classes=2, dropout=0.0, pooling="mean"):
        super(LSTM_ChaosClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size = x.size(0)

        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))

        if self.pooling == "last":
            x = lstm_out[:, -1, :]
        elif self.pooling == "mean":
            x = torch.mean(lstm_out, dim=1)
        elif self.pooling == "max":
            x, _ = torch.max(lstm_out, dim=1)
        else:
            raise ValueError("pooling must be one of ['last', 'mean', 'max']")

        x = self.fc1(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nhead, device):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.attn_head_dim = hidden_dim // nhead
        self.device = device

        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)

        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, query, key, value):
        B, N, _ = query.size()

        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        Q = Q.view(B, N, self.nhead, self.attn_head_dim).transpose(1, 2)
        K = K.view(B, N, self.nhead, self.attn_head_dim).transpose(1, 2)
        V = V.view(B, N, self.nhead, self.attn_head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(self.device )
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)

        output = self.out_linear(attn_output)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, device):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(hidden_dim, nhead, device)
        self.ffn = PositionwiseFeedForward(hidden_dim)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_output = self.attn(x, x, x)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x


class Transformer_ChaosClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, nhead=4, num_layers=1, num_classes=2, device=None, max_len=None):
        super(Transformer_ChaosClassifier, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(hidden_dim, nhead, device) for _ in range(num_layers)])

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)
        x = self.global_pooling(x).squeeze(-1)
        x = self.fc(x)
        return x
