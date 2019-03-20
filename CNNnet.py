import torch
import torch.nn as nn


class CNNnet(nn.Module):
    def __init__(self):
        super().__init__()


        #self.embed = nn.Embedding()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100,kernel_size=(3, n),bias = True)
        self.conv2 = nn.Conv2d(1, 100, (4,n))
        self.conv3 = nn.Conv2d(1, 100,(5,n))
        self.f1 = nn.Linear(150, class_num)

    def forward(self, x, embedding_model):
        embed_x = embedding_model(x)
        out1 = nn.MaxPool1d(nn.ReLU(self.conv1(embed_x)), n-2) #(n,k)->((n-3+1), 1)*100->(1,100)
        out2 = nn.MaxPool1d(nn.ReLU(self.conv2(embed_x)), n-3) #(n,k)->((n-4+1), 1)*100->(1,100)
        out3 = nn.MaxPool1d(nn.ReLU(self.conv3(embed_x)), n-4) #(n,k)->((n-5+1), 1)*100->(1,100)
        out =  torch.cat((out1, out2, out3), 1) # (1, 300)
        out = nn.Dropout(out) # (1, 150)
        out = self.f1(out) # (1, class_num)
        return out





