import sys
import os
from os.path import join
import time
import datetime
import itertools

import numpy as np
import scipy
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk import sent_tokenize
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

import torch

from features.embedding_features import ExtractWordEmbeddings
from models.lstm import LSTMClassifier


class TenDimensionsLSTM:
    def __init__(self, models_dir, embeddings_dir):
        # where the saved models are stored
        self.models_dir = models_dir
        # where the embedding files are stored
        self.embeddings_dir = embeddings_dir

        # text tokenizer
        self.tokenize = TweetTokenizer().tokenize

        # the 10 dimensions
        self.dimensions = ['knowledge', 'power', 'respect', 'trust', 'social_support',
                           'romance', 'similarity', 'identity', 'fun', 'conflict']
        self.dimensions2name = {'social_support': 'support', 'romance': 'romance', 'identity': 'identity',
                                'similarity': 'similarity', 'conflict': 'conflict', 'fun': 'fun',
                                'knowledge': 'knowledge', 'respect': 'status', 'trust': 'trust', 'power': 'power'}
        self.name2dimensions = {'support': 'social_support', 'romance': 'romance', 'identity': 'identity',
                                'similarity': 'similarity', 'conflict': 'conflict', 'fun': 'fun',
                                'knowledge': 'knowledge', 'status': 'respect', 'trust': 'trust', 'power': 'power'}

        # load the embeddings
        em_glove = ExtractWordEmbeddings('glove', emb_dir=self.embeddings_dir)
        em_word2vec = ExtractWordEmbeddings('word2vec', emb_dir=self.embeddings_dir)
        em_fasttext = ExtractWordEmbeddings('fasttext', emb_dir=self.embeddings_dir)

        # initialize the dimension-specific models
        self.dimension2model = {}
        self.dimension2embedding = {}
        for dim in self.dimensions:
            is_cuda = False  # True
            model = LSTMClassifier(embedding_dim=300, hidden_dim=300)
            if is_cuda:
                model.cuda()
            model.eval()
            for modelname in os.listdir(self.models_dir):
                if ('-best.lstm' in modelname) & (dim in modelname):
                    best_state = torch.load(join(self.models_dir, modelname), map_location='cpu')
                    model.load_state_dict(best_state)
                    if 'glove' in modelname:
                        em = em_glove
                    elif 'word2vec' in modelname:
                        em = em_word2vec
                    elif 'fasttext' in modelname:
                        em = em_fasttext
                    self.dimension2model[dim] = model
                    self.dimension2embedding[dim] = em
                    break

    def compute_score(self, texts, dimensions):
        if dimensions is None:
            dimensions = self.dimensions
        else:
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(dimensions, str):
                dimensions = [dimensions]
            dimensions = [self.name2dimensions[d] for d in dimensions]

        results = []
        for text in texts:
            text_results = {}
            for dimension in dimensions:
                #try:
                classifier = self.dimension2model[dimension]
                em = self.dimension2embedding[dimension]
                input_ = em.obtain_vectors_from_sentence(self.tokenize(text), True)
                input_ = torch.tensor(input_).float().unsqueeze(0)
                # if is_cuda:
                #    input_ = input_.cuda()
                output = classifier(input_)
                score = torch.sigmoid(output).item()
                text_results[self.dimensions2name[dimension]] = score
            #except:
                #    text_results[dimension] = 0.0
            results.append(text_results)
        return results

def test():
    # Load model
    models_dir = 'lstm_trained_models'
    embeddings_dir = 'C:\\Users\\lajel\\embeddings'  # change urls to embeddings dir
    model = TenDimensionsLSTM(models_dir=models_dir, embeddings_dir=embeddings_dir)
    print('Model loaded')

    # Run model
    sentences = ['How are you? I really hope you feel better now',
                 'What have you just said? Your opinions are very silly, I do not want to listen anymore',
                 'The Unix operating system was created in November 1971',
                 'All our employees know what they do, they must be trusted',
                 'Oh man, I laughed so hard at his joke that I spit my coffee',
                 'This is a tradition typical of my people',
                 'I desire you, my love',
                 'Thank to all employees, they have done a fantastic job!'
                 ]
    # you can give in input both texts or a list of texts
    scores = model.compute_score(sentences, dimensions=None)  # dimensions = None extracts all dimensions
    for sent, score in zip(sentences, scores):
        print(f'{sent} -- {score}')
