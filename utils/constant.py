from __future__ import absolute_import
import torch

# entitiy_to_index = torch.load('../data/processed_data/entity_to_index.pt')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model
hidden_size = 128
embed_size = 64
maxlen = 32
epoch = 100
batch_size = 64
dropout = 0.5

# LSTM
lstm_layers=2

# CNN for character embed
channel_in = 1
channel_out = 32
kernel_sizes = [2,3,4]

# Optimizer
learning_rate = 5e-4
gradient_accumulation_steps = 2
warmup_proportion = 0.1
warmup_steps = 100

# Index
max_char_len = 5
# pad_idx = 1      # tok.convert_tokens_to_ids('[PAD]')
# pad_label = 21   # entitiy_to_index['[PAD]']
# o_label = 20     # entitiy_to_index['O']
