# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:28:16 2019

@author: Administrator
"""

#!/usr/bin/python3
# -*- coding: UTF-8 -*-


from argparse import ArgumentParser
import logging
import math
import random
import sys
import h5py
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from vocab import Vocab
from log_timer import LogTimer
from gensim.models import FastText 




def processing2(dataset, PAD_id):
''' To transfer the data as list format and remove PAD tokens'''    
    uttras = []
    for uttra in dataset:
        sentences = []
        for word in uttra:
            if word != PAD_id:
                sentences.append(word-1)# transfer the min value of word indexs from 1 to 0  
        uttras.append(sentences)
    return uttras

           
class RnnLm(nn.Module):
    """ A language model RNN with GRU or LSTM layer(s). """

    def __init__(self, vocab_size, args):
        super(RnnLm, self).__init__()
        self.embedding_dim = args.embedding_dim  
        self.batch_size = args.batch_size    
        self.layers = args.layers        
        self.dropout = args.gru_dropout      
        self.hidden_dim =  args.gru_hidden
        self.embedding_size = args.embedding_dim
        
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.layers,
                          dropout=self.dropout)
#        self.gru = nn.LSTM(embedding_dim, hidden_dim, gru_layers,
#                          dropout=dropout)
        self.fc1 = nn.Linear(self.hidden_dim, vocab_size)
        self.init_weight()
        logging.debug("Net:\n%r", self)
        
        
        
    def init_weight(self):
        
        init_range = 0.08
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.embedding.weight.data.uniform_(-init_range, init_range)
#    def get_embedded(self, word_indexes):
#        if self.tied:
#            return self.fc1.weight.index_select(0, word_indexes)
#        else:
#            return self.embedding(word_indexes)

    def forward(self, packed_sents, hidden_init):
        """ Takes a PackedSequence of sentences tokens that has T tokens
        belonging to vocabulary V. Outputs predicted log-probabilities
        for the token following the one that's input in a tensor shaped
        (T, |V|).
        """
        embedded_sents = nn.utils.rnn.PackedSequence(
            self.embedding(packed_sents.data), packed_sents.batch_sizes)
        out_packed_sequence, h_n = self.gru(embedded_sents, hidden_init)
        out = self.fc1(out_packed_sequence.data)
        return F.log_softmax(out, dim=1), h_n
    def init_hidden(self, batch_size):
        ''' initialize the hidden state with shape
        (num_layers * num_directions, batch, hidden_size)'''
        
        return torch.FloatTensor( self.layers, batch_size, self.embedding_size).uniform_(-0.08, 0.08)

def batches(data, batch_size):
    """ Yields batches of sentences from 'data', ordered on length. """
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]


def step(model, hidden_init, utterances, device):
    """ Performs a model inference for the given model and sentence batch.
    Returns the model output, total loss, target outputs and hidden state. """
    #Packs a list of variable length Tensors without padding
    x = nn.utils.rnn.pack_sequence([u[:-1] for u in utterances])
    y = nn.utils.rnn.pack_sequence([u[1:] for u in utterances])
    if device.type == 'cuda':
        x, y = x.to(device), y.to(device)
    
    out, h_n = model(x, hidden_init)
    loss = F.nll_loss(out, y.data)
    return out, loss, y, h_n


def train_epoch(data, model, optimizer, args, device, id2word2):
    """ Trains a single epoch of the given model. """
    model.train()
    log_timer = LogTimer(5)
    for batch_ind, batched_uts in enumerate(batches(data, args.batch_size)):
        model.zero_grad()        
        hidden_init = model.init_hidden(len(batched_uts)).to(device)
        
        for index in batched_uts[0]:
            print(id2word2[index.item()], end= ' ') # print an uttrance
            
        out, loss, y, h_n = step(model, hidden_init, batched_uts, device)
        loss.backward()
        optimizer.step()
        if log_timer() or batch_ind == 0:
            # Calculate perplexity.
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            perplexity = 2 ** prob.log2().neg().mean().item()
            logging.info("\tBatch %d, loss %.3f, perplexity %.2f",
                         batch_ind, loss.item(), perplexity)


def evaluate(data, model, batch_size, device):
    """ Perplexity of the given data with the given model. """
    model.eval()
    with torch.no_grad():
        entropy_sum = 0
        word_count = 0
        for sents in batches(data, batch_size):
            hidden_init = model.init_hidden(len(sents)).to(device)
            out, _, y, h_n = step(model, hidden_init, sents, device)
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.log2().neg().sum().item()
            word_count += y.data.shape[0]
    return 2 ** (entropy_sum / word_count)


def parse_args(args):
    argp = ArgumentParser(description=__doc__)
    argp.add_argument("--logging", choices=["INFO", "DEBUG"],
                      default="INFO")

    argp.add_argument("--embedding-dim", type=int, default=1000,
                      help="Word embedding dimensionality")
    argp.add_argument("--untied", action="store_true",
                      help="Use untied input/output embedding weights")
    argp.add_argument("--gru-hidden", type=int, default=1000,
                      help="GRU gidden unit dimensionality")
    argp.add_argument("--layers", type=int, default=2,
                      help="Number of GRU layers")
    argp.add_argument("--gru-dropout", type=float, default=0,
                      help="The amount of dropout in GRU layers")

    argp.add_argument("--epochs", type=int, default=40)
    argp.add_argument("--batch-size", type=int, default=2)
    argp.add_argument("--lr", type=float, default=0.01,
                      help="Learning rate")

    argp.add_argument("--no-cuda", action="store_true")
    return argp.parse_args(args)


def main(args=sys.argv[1:]):
    args=sys.argv[1:]
    args = parse_args(args)
    logging.basicConfig(level=args.logging)
    
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    reddit_path = 'data/Aligned-Dataset/reddit.h5'
    
    dictionary_path = 'data/Aligned-Dataset/dictionary.json'
    
    reddit = h5py.File(reddit_path, 'r')
    
    # Load data now to know the whole vocabulary when training model.
    with open(dictionary_path, 'r') as f:
    
        dictionary = json.load(f)
        id2word = dictionary['id2word']
        id2word = {int(key): id2word[key] for key in id2word}
        word2id = dictionary['word2id']
        f.close()
    id2word2 = dict(zip(np.array(list(id2word.keys())) - 1, id2word.values())) # to make the keys start from 0 instead of 1

    PAD_id = word2id['<PAD>']
    EOS_token = word2id['<end>']
    new_train = processing2(reddit['train'],PAD_id)
    new_valid = processing2(reddit['validate'],PAD_id)
    new_test = processing2(reddit['test'],PAD_id)
    
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                              else "cuda")
#    device = torch.device("cpu")
    
    model = RnnLm(len(id2word), args).to(device)

#    if torch.cuda.device_count() > 1:
#        print("Let's use", torch.cuda.device_count(), "GPUs!")
#        model = nn.DataParallel(model)   
#        model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum =  0.9 )
    
   
    
    for epoch_ind in range(args.epochs):
        logging.info("Training epoch %d", epoch_ind)
        
        train_epoch(new_train, model, optimizer, args, device, id2word2)

        logging.info("Validation perplexity: %.1f",
                     evaluate(new_valid, model, args.batch_size, device))
        torch.save(model, './model/modelLang2.pt')
        
    logging.info("Test perplexity: %.1f",
                 evaluate(new_test, model, args.batch_size, device))
    torch.save(model, './model/modelLang2.pt')


if __name__ == '__main__':
   
    
    
    main()
