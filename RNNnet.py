import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNnet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()


        self.gru = nn.GRU(input_size, hidden_size, num_layers = 2)
        self.softmax = nn.Softmax()

    def forward(self, x, embedding_model,features):

        embed_x = embedding_model(x)
        out, hidden = self.gru(embed_x)
        ht = hidden @ features # couple
        ht = F.softmax(ht,dim=1)
        return ht







