import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod

def get_policies(in_size, out_size, config):
    policies = {
        'mlp': MLPPolicy,
        'rnn': RNNPolicy,
        'transformer': TransformerPolicy
    }
    policy_a = policies[config['policy_a_arch']](in_size, out_size, **config['arch_a'])
    policy_b = policies[config['policy_b_arch']](in_size, out_size, **config['arch_b'])
    return policy_a, policy_b

class MLPPolicy(nn.Module):
    def __init__(self, input_size, output_size,  hidden_size=32):
        super(MLPPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x.flatten())
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

class RNNPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(RNNPolicy, self).__init__()
        self.rnn = nn.RNN(64, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # No softmax

    def forward(self, x):
        #print(x.shape)
        #if len(x.shape) == 3:
        #    x = x.reshape(30, 7, 64)
        #else: x = x.reshape(30, 7, 64)
        #print(x.shape)
        if isinstance(x, torch.Tensor):
            x = x.reshape(x.shape[0], 7, 64)
        else:
            x = x.reshape(int(prod(x.shape[:-2])), 7, 64)
            x = torch.tensor(x, dtype=torch.float32)
            
        out, _ = self.rnn(x)
        logits = self.fc(out[:, -1, :])
        #print(logits)
        return torch.distributions.Categorical(logits=logits)

class TransformerPolicy(nn.Module):
    def __init__(self, input_size, output_size,
        hidden_size=32, nhead=4, num_encoder_layers=2, dim_feedforward=32):
        super(TransformerPolicy, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
            nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(transformer_layer,
            num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        logits = self.fc_out(x[:, -1, :])
        return torch.distributions.Categorical(logits=logits)
