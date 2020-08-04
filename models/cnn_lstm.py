import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .crf import CRF
import utils.constant as config


class CharCNNEmbedding(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_sizes, char_vocab_size, embed_size):
        super(CharCNNEmbedding, self).__init__()
        
        self.char_embed = nn.Embedding(char_vocab_size, embed_size, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, channel_out, (ks, embed_size)) for ks in kernel_sizes])
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, char):
        """
        Input:
            char : torch.tensor of size (batch, max_word_len, max_char_len)
        """

        char_embedding = []
        for i in range(char.size(1)):
            char_idx = char[:, i]  # (batch, max_char_len)
            char_embed = self.char_embed(char_idx).unsqueeze(1) # (batch, 1, max_char_len, embed_size)
            char_conv = [F.relu(conv(char_embed)).squeeze(3) for conv in self.convs] # (batch, ch_out, i) where i=4,3,2,1
            char_pool = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in char_conv] # (batch, channel_out)
            char_pool_concat = torch.cat(char_pool, 1) # (batch, channel_out*len(kernel_sizes))
            char_pool_concat = char_pool_concat.unsqueeze(1) # (batch, 1, channel_out*len(kernel_sizes))
            char_embedding.append(char_pool_concat)

        # conver to torch tensor
        char_embedding = torch.cat(char_embedding, 1) # (batch, max_Word_len, channel_out*len(kernel_sizes))

        return char_embedding
            
                

class CNNBiLSTM(nn.Module):
    def __init__(self, config, num_classes, vocab_size, char_vocab_size, pos_vocab_size, word2vec=None, use_crf=False):
        #kernel_num=128, kernel_sizes=[2,3,4],
        super(CNNBiLSTM, self).__init__()
        
        embed_size = config.embed_size
        hidden_size = config.hidden_size
        drouput = config.dropout
        lstm_layers = config.lstm_layers
        
        channel_in = config.channel_in
        channel_out = config.channel_out
        kernel_sizes = config.kernel_sizes
        
        self.word2vec = word2vec
        self.use_crf=use_crf
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_embed = nn.Embedding(pos_vocab_size, embed_size, padding_idx=0)
        self.char_embed = CharCNNEmbedding(channel_in, channel_out, kernel_sizes, char_vocab_size, embed_size)
        
        if self.word2vec is not None:
            self.word_embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.word_embed.weight = torch.nn.parameter.Parameter(torch.Tensor(word2vec))
            self.word_embed.weight.requires_grad = False
            self.lstm = nn.LSTM((channel_out * len(kernel_sizes) + 2*embed_size + embed_size),
                                hidden_size, num_layers, dropout=0.6, batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM((channel_out * len(kernel_sizes) + embed_size + embed_size),
                                hidden_size, lstm_layers, dropout=drouput, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(drouput)
        self.linear = nn.Linear(2 * hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, x, x_char, x_pos):
        
        # Embedd
        x_word_embedding = self.embed(x)            # (batch,max_word_len, embed_size)
        x_pos_embedding = self.pos_embed(x_pos)     # (batch,max_word_len, embed_size)
        x_char_embedding = self.char_embed(x_char)  # (batch,max_word_len, channel_out * len(kernel_sizes))
 
        # Concat
        embed_concat = torch.cat((x_word_embedding, x_char_embedding, x_pos_embedding), 2) # (b, word, 2*embed_size + ch*len(ks))
        embed_concat = self.dropout(embed_concat)
        
        # LSTM and Linear
        lstm_out, state = self.lstm(embed_concat)
        linear_out = self.linear(lstm_out)
        
        if self.use_crf:
            log_likelihood = self.crf(linear, tags)
            predict_tag = self.crf.decode(linear)
            return log_likelihood, predict_tag          
        else:
            return self.dropout(linear_out)
    

#     def sample(self, x, x_char, x_pos, x_lex_embedding, lengths):

#         x_word_embedding = self.embed(x)  # (batch,words,word_embedding)
#         trainable_x_word_embedding = self.trainable_embed(x)

#         char_output = []
#         for i in range(x_char.size(1)):
#             x_char_embedding = self.char_embed(x_char[:, i]).unsqueeze(1)  # (batch,channel_input,words,word_embedding)

#             h_convs1 = [F.relu(conv(x_char_embedding)).squeeze(3) for conv in self.convs1]
#             h_pools1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in
#                         h_convs1]  # [(batch,channel_out), ...]*len(kernel_sizes)
#             h_pools1 = torch.cat(h_pools1, 1)  # 리스트에 있는걸 쭉 쌓아서 Tensor로 만듬!
#             h_pools1 = self.dropout(h_pools1)  # (N,len(Ks)*Co)
#             out = h_pools1.unsqueeze(1)  # 단어단위 고려
#             char_output.append(out)

#         char_output = torch.cat(char_output, 1)  # 단어 단위끼리 붙이고 # torch.cat((h_pools1, h_lexicon_pools1), 1)

#         x_pos_embedding = self.pos_embed(x_pos)

#         enhanced_embedding = torch.cat((char_output, x_word_embedding, trainable_x_word_embedding, x_pos_embedding), 2)  # 임베딩 차원(2)으로 붙이고
#         enhanced_embedding = self.dropout(enhanced_embedding)
#         enhanced_embedding = torch.cat((enhanced_embedding, x_lex_embedding), 2)

#         output_word, state_word = self.lstm(enhanced_embedding)
#         logit = self.fc1(output_word)

#         return logit