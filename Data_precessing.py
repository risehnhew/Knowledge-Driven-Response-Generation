# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:15:35 2019

@author: Administrator
"""
import json
import numpy as np
import h5py

reddit_path = 'C:/Users/Administrator/Python_code/Aligning-Reddit-and-Wikipedia-master/Aligned-Dataset/reddit.h5'
wikipedia_path = 'C:/Users/Administrator/Python_code/Aligning-Reddit-and-Wikipedia-master/Aligned-Dataset/wikipedia.h5'
dictionary_path = 'C:/Users/Administrator/Python_code/Aligning-Reddit-and-Wikipedia-master/Aligned-Dataset/dictionary.json'

reddit = h5py.File(reddit_path, 'r')
wikipedia = h5py.File(wikipedia_path, 'r')

with open(dictionary_path, 'r') as f:
    #dictionary = json.load(f, 'utf-8')
    dictionary = json.load(f)
    id2word = dictionary['id2word']
    id2word = {int(key): id2word[key] for key in id2word}
    word2id = dictionary['word2id']
    f.close()

def getAligned(index, dataset = 'train', reddit, wikipedia):
    if dataset == 'train' or dataset == 'test' or dataset == 'validate':
        if index < len(reddit[dataset]):
            i = 0
            sequence = ''
            while reddit[dataset][index][i + 1] != word2id['<PAD>']:
                if reddit[dataset][index][i] == word2id['<end>'] or reddit[dataset][index][i] == word2id['<eot>']:
                    sequence = sequence + id2word[reddit[dataset][index][i]] + '\n'
                else:
                    sequence = sequence + id2word[reddit[dataset][index][i]] + ' '
                i += 1
            sequence = sequence + id2word[reddit[dataset][index][i]]
            sentences = []
            for j in range(0, 20):
                i = 0
                sentences.append('')
                while wikipedia[dataset][index * 20 + j][i + 1] != word2id['<PAD>']:
                    sentences[j] += id2word[wikipedia[dataset][index * 20 + j][i]]+ ' '
                    i += 1
                sentences[j] += id2word[wikipedia[dataset][index * 20 + j][i]]

            print ('Number (%d) Sequence of Comments from the (%s) Set\n' % (index, dataset.title()))
            print (sequence)
            print ('\n\nWikipedia Sentences for the Number (%d) Sequence of Comments from the (%s) Set\n' % (index, dataset.title()))

            print ('\n'.join(sentences))
        else:
            print ('The index exceeds the available examples in the %s Set.' % (dataset.title()))
            print ('Pick an index between 0 and %d for the %s Set.' % (len(reddit[dataset]) - 1, dataset.title()))
    else:
        print('The available options for the dataset variable are: train, validation and test.')
    return {'reddit': sequence, 'wikipedia':sentences}

pairs = getAligned(257, dataset = 'validate')