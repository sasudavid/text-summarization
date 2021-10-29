from __future__ import unicode_literals, print_function, division
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from io import open
from rouge_score import rouge_scorer
import nltk
#from nltk.corpus import stopwords

import ast
import chardet
import pandas as pd
import numpy as np
import json
import os
import glob

import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import networkx as nx

np.seterr(divide='ignore', invalid='ignore')


contraction_map = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                   "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                   "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",

                           "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}



def nltk_stop_words(filename):
    inputfile = open(filename, "r")
    stopwords = []
    for line in inputfile:
        stopwords.append(line.rstrip('\n'))

    return stopwords



def read_article(file_text):
    article = file_text.split(". ")
    sentences = []

    for i in range(len(article)):
        preprocessed_sentence = article[i].replace('\n','')
        sentences.append(preprocessed_sentence)
    
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_text, top_n=1):
    stop_words = nltk_stop_words("english_stop_words")
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_text)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    try:
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph, max_iter=100000)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True) 
        #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

        for i in range(top_n):
          summarize_text.append("".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize text

        output = summarize_text

    except:
        output = ["Could not generate summary"]


    return output



# let's begin

def obtain_articles(filename):
    dataset = []
    target = []

    precision = 0
    recall = 0
    fmeasure = 0
    #open the file
    inputfile = open(filename, 'r')
    for line in inputfile:
        json_line = json.loads(line)
        source_data = json_line["text"]
        target_data = json_line["title"]
        dataset.append(source_data)
        target.append(target_data)

    for i in range(len(dataset)):
        output_text = generate_summary(dataset[i])
        output_text = output_text[0]
        
        if output_text != "Could not generate summary":
            scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
            score = scorer.score(target[i], output_text)

            #print out the rouge score of the predicted sequence
            precision += score['rouge1'].precision
            recall += score['rouge1'].recall
            fmeasure += score['rouge1'].fmeasure

    average_precision = precision/len(dataset)
    average_recall = recall/len(dataset)
    average_fmeasure = fmeasure/len(dataset)

    print("average precision: ",average_precision)
    print("average recall: ",average_recall)
    print("average fmeasure: ",average_fmeasure)

    return


obtain_articles('train.json')
























