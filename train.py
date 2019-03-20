import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataloader
import json, h5py
from RNNnet import RNNnet
from CNNnet import CNNnet
from Data_precessing import getAligned
from gensim.models import KeyedVectors
import argparse

parser = argparse.ArgumentParser(description= 'This is a Knowledge Driven Response Generation implementation')
parser.add_argument('--reddit_path',  type = str, default ='C:/Users/Administrator/Python_code/Aligning-Reddit-and-Wikipedia-master/Aligned-Dataset/reddit.h5' )
parser.add_argument('--wikepedia_path', type = str, default = 'C:/Users/Administrator/Python_code/Aligning-Reddit-and-Wikipedia-master/Aligned-Dataset/wikipedia.h5')
parser.add_argument('--dictionary_path', type = str, default='C:/Users/Administrator/Python_code/Aligning-Reddit-and-Wikipedia-master/Aligned-Dataset/dictionary.json')
parser.add_argument('--w2v_path', type= str, default= 'H:/w2vdata/GoogleNews-vectors-negative300.bin')
parser.add_argument('--pre_data_path', type= str, default= 'C:/Users/Administrator/Python code/Aligning-Reddit-and-Wikipedia-master/Data/Wikipedia/Sentences.json')

#parser.add_argument('--lr')
parser.add_argument('--pretrain_epoach',type = int, default= 10 )

reddit_path = 'C:/Users/Administrator/Python_code/Aligning-Reddit-and-Wikipedia-master/Aligned-Dataset/reddit.h5'
wikipedia_path = 'C:/Users/Administrator/Python_code/Aligning-Reddit-and-Wikipedia-master/Aligned-Dataset/wikipedia.h5'
dictionary_path = 'C:/Users/Administrator/Python_code/Aligning-Reddit-and-Wikipedia-master/Aligned-Dataset/dictionary.json'

args = parser.parse_args()

reddit = h5py.File(args.reddit_path, 'r')
wikipedia = h5py.File(args.wikipedia_path, 'r')

w2v_model = KeyedVectors.load_word2vec_format(args.w2v_path, binary=True)

with open(dictionary_path, 'r') as f:
    #dictionary = json.load(f, 'utf-8')
    dictionary = json.load(f)
    id2word = dictionary['id2word']
    id2word = {int(key): id2word[key] for key in id2word}
    word2id = dictionary['word2id']
    f.close()

pairs = getAligned(257, dataset = 'validate')

with open(args.pre_data_path, 'r') as pre_f:
    pre_sentences = json.load(pre_f, encoding='utf-8')

pre_labels = pre_sentences.keys()









loss_function = nn.CrossEntropyLoss()
model_CNN = CNNnet()
model_RNN = RNNnet()
CNN_optimizer = torch.optim.adam(model_CNN.parameters())
model_CNN.train()




for epoch in range(1, args.pretrain_epoch + 1):
    output = model_CNN()

    loss = loss_function(output, target_pre)
    loss.backword()
    optimizer.step()



for epoch in range(10):
    features = model_CNN(reddit, w2v_model)
    sentences = model_RNN(wikipedia, w2v_model)