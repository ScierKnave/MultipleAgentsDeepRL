
def get_policies(in_size, out_size, config):
    policy_a = config['policy_a_arch'](in_size, out_size, **config['arch_a'])
    policy_b = config['policy_b_arch'](in_size, out_size, **config['arch_b'])
    return policy_a, policy_b

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPolicy(nn.Module):
    def __init__(self, input_size, output_size,  hidden_size=128):
        super(SimpleMLPPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

class RNNPolicy(nn.Module):
    def __init__(self, input_size, output_size,hidden_size=32):
        super(SimpleRNNPolicy, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # No softmax

    def forward(self, x):
        out, _ = self.rnn(x)
        logits = self.fc(out[:, -1, :])
        return torch.distributions.Categorical(logits=logits)

class TransformerPolicy(nn.Module):
    def __init__(self, input_size, output_size,
        hidden_size=32, nhead=4, num_encoder_layers=2, dim_feedforward=32):
        super(SimpleTransformerPolicy, self).__init__()
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
