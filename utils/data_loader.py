from __future__ import absolute_import, division, print_function, unicode_literals
import pdb
import numpy as np
import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from . import constant as config
device = config.device


class NERDataset(Dataset): 
    def __init__(self,dtype):
        self.token = torch.load('./data/processed_data/{}_token_idx.pt'.format(dtype))
        self.char = torch.load('./data/processed_data/{}_char_idx.pt'.format(dtype))
        self.pos = torch.load('./data/processed_data/{}_pos_idx.pt'.format(dtype))
        self.label = torch.load('./data/processed_data/{}_label.pt'.format(dtype))
        
        assert len(self.token)==len(self.char)==len(self.pos)==len(self.label)
        self.length = len(self.label)
        
    def __getitem__(self, idx):
        return self.token[idx], self.char[idx], self.pos[idx], self.label[idx]
    
    def __len__(self):
        return self.length
    
    
def pad_collate(batch):
    
    def to_tensor(x):
        return torch.from_numpy(x).long().to(device)
    
    def _pad(sequence, maxlen):
        return pad_sequences(sequence, maxlen=maxlen, value=0, 
                             padding='post', dtype='long', truncating='post')
    
    def _label_pad(sequence, maxlen):
        return pad_sequences(sequence, maxlen=maxlen, value=0, 
                         padding='post', dtype='long', truncating='post')

    def _char_pad(char, maxlen1, maxlen2):
        maxlen1 = max(maxlen1, config.max_char_len)
        padded = []
        for c in char:
            char_pad = _pad(tuple(c), maxlen1)
            residual = np.zeros([maxlen2-char_pad.shape[0], char_pad.shape[1]])
            char_pad = np.vstack([char_pad, residual])
            padded.append(char_pad)
        # pdb.set_trace()
        return np.array(padded)
    
    (token, char, pos, label) = zip(*batch)
    token_len = [len(t) for t in token]
    char_len  = [len(c) for c in char]
    
    # Padding
    token_pad = to_tensor(_pad(token, max(token_len)))
    char_pad = to_tensor(_char_pad(char, max(char_len), max(token_len)))
    pos_pad = to_tensor(_pad(pos, max(token_len)))
    label_pad = to_tensor(_pad(label, max(token_len))) 
    
    return token_pad, char_pad, pos_pad, label_pad

    # Packing
#     input_length = torch.LongTensor([torch.max(token_pad[i, :].data.nonzero())+1 for i in range(token_pad.size(0))])
#     sorted_input_length, sorted_idx = input_length.sort(0, descending=True)
#     token_pad, char_pad, pos_pad, label_pad = map(lambda x: x[sorted_idx], [token_pad, char_pad, pos_pad, label_pad])
    
#     return token_pad, char_pad, pos_pad, label_pad, sorted_input_length.tolist()

