from __future__ import unicode_literals, print_function, division
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
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# initializing a list to hold the source data from the files
dataset = []

# initialize a list to hold the target values of the source data from the files
target = []


# define a contraction hashmap to help pre-process the text data
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

# extract the data from the files and place it in python lists


def extractor(filepath):
    inputfile = open(filepath, 'r')
    for line in inputfile:
        json_line = json.loads(line)
        source_data = json_line["text"]
        target_data = json_line["title"]
        dataset.append(source_data)
        target.append(target_data)

    inputfile.close()
    return

# load all of the data files in the directory


def dataloader(data_directory):
    for i in os.listdir(data_directory):
        data_file_path = os.path.join(data_directory, i)
        if '.json' in data_file_path:
            extractor(data_file_path)

    return

# obtain the english stop words from nltk and take those words from the dataset


def nltk_stop_words(filename):
    inputfile = open(filename, "r")
    stopwords = []
    for line in inputfile:
        stopwords.append(line.rstrip('\n'))

    return stopwords

# preprocess each line of text in the dataset and the target lists


def preprocess(text, nltk_english_stop_words):
    # convert the text to lowercase
    lowercase_text = text.lower()
    # split the text
    split_text = lowercase_text.split()
    # remove apostrophe endings from the text using regular expressions
    reconstructed_text = []

    for j in range(len(split_text)):
        if split_text[j] in contraction_map:
            split_text[j] = contraction_map[split_text[j]]

        # remove apostrophe endings from the text
        split_text[j] = split_text[j].replace("'s", '')

        # use regular expressions to remove parentheses outside a word
        split_text[j] = re.sub(r'\(.*\)', '', split_text[j])

        # use regular expressions to remove punctuations
        split_text[j] = re.sub(r'[^a-zA-Z0-9. ]', '', split_text[j])

        # add a space character before and after a full stop
        split_text[j] = re.sub(r'\.', ' . ', split_text[j])

        # remove stop words from the text
        if split_text[j] not in nltk_english_stop_words:
            reconstructed_text.append(split_text[j])

    reconstructed_text = " ".join(reconstructed_text)

    return reconstructed_text


# run each sentence in the dataset lists in the preprocess function
def run_preprocess(data, stop_words):
    for i in range(len(data)):
        data[i] = preprocess(data[i], stop_words)

    return data

# ensure that the texts in the source dataset have a length of 600 and the texts in the target dataset have a length of 30


def trim_data(source_data, target_data, max_len_source=600, max_len_target=30):
    trimmed_source_data = []
    trimmed_target_data = []

    for i in range(len(source_data)):
        if len(source_data[i].split()) <= max_len_source and len(target_data[i].split()) <= max_len_target:
            trimmed_source_data.append(source_data[i])
            trimmed_target_data.append(target_data[i])

    temporary_dataframe = pd.DataFrame(
        {'source_data': trimmed_source_data, 'summary': trimmed_target_data})
    new_dataframe = temporary_dataframe[temporary_dataframe['summary'].str.strip(
    ).astype(bool)]
    dataframe = new_dataframe[new_dataframe['source_data'].str.strip().astype(
        bool)]

    return dataframe

# create a class to help with one-hot encoding


class Lang:
    # initialise the instance variables of the class
    def __init__(self, type, text_data):
        self.text_data = text_data
        if type == "target":
            self.marked_sentences = self.mark_target_sentences(self.text_data)
        else:
            self.marked_sentences = text_data
        self.SOS_token = 0
        self.EOS_token = 1
        self.combined_list = self.combine_words(self.marked_sentences)
        self.total_num_words = len(self.combined_list)
        self.word2idx = self.word2index(self.combined_list)
        self.idx2word = self.index2word(self.word2idx)
        self.wordcount = self.word2count(self.combined_list)

    # mark the start and the end of sentences with the start of sentence token and the end of sentence token

    def mark_target_sentences(self, text_data):
        for i in range(len(text_data)):
            text_data[i] = '<SOS>'+' '+text_data[i]+' '+'<EOS>'

        return text_data

    # combine the words in the source data and the target data into a list

    def combine_words(self, text_data):
        combined_data_list = []

        for sentence in text_data:
            split_sentence = sentence.split()
            combined_data_list += split_sentence

        return combined_data_list

    # create a dictionary with the index of the word as the key and the value representing the word as the value

    def word2index(self, combined_data_list):
        word2idx = {'<SOS>': 0, '<EOS>': 1}
        for i in range(len(combined_data_list)):
            if combined_data_list[i] not in word2idx:
                if i == 0:
                    word2idx[combined_data_list[i+2]] = i + 2
                elif i == 1:
                    word2idx[combined_data_list[i+1]] = i + 1
                else:
                    word2idx[combined_data_list[i]] = i

        return word2idx

    # create a dictionary with the word as the key and the index of the word's first appearance as the value
    def index2word(self, data_dict):
        word2index_dict = data_dict
        idx2word = {index: word for word, index in word2index_dict.items()}

        return idx2word

    # create a dictionary with the words as the key and the count of the words in the
    def word2count(self, combined_data_list):
        word2count_dict = {}
        for word in combined_data_list:
            if word not in word2count_dict:
                word2count_dict[word] = 1
            elif word in word2count_dict:
                word2count_dict[word] = word2count_dict[word] + 1

        return word2count_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a class for the encoder Gated Recurrent Unit Neural Network


class Encoder(nn.Module):
    # initialise the neural network
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size).to(device='cuda')
        self.gru = nn.GRU(hidden_size, hidden_size).to(device='cuda')

    # perform a forward pass through the network
    def forward(self, input_size, hidden):
        output = self.embedding(input_size).view(1, 1, -1).to(device = 'cuda')
        output, hidden = self.gru(output, hidden)

        return output, hidden

    # initialise the hidden layer of the network

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device).to(device = 'cuda')

# keep track of the maximum number of words in a sequence
#MAX_LENGTH = 30


# create a class for the decoder Gated Recurrent Network
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_prob=0.2, max_length=600):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size).to(device='cuda')

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length).to(device='cuda')
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size).to(device='cuda')

        self.dropout = nn.Dropout(self.dropout_prob).to(device='cuda')
        self.gru = nn.GRU(self.hidden_size, self.hidden_size).to(device='cuda')
        self.out = nn.Linear(self.hidden_size, self.output_size).to(device='cuda')

    def forward(self, input_size, hidden, encoder_outputs):
        # ensure that the input vectors are embedded to reduce the dimensions of the input vectors
        embedded = self.embedding(input_size).view(1, 1, -1).to(device = 'cuda')
        # when dropout is performed there is the probability of the embedding to be 'zeroed'.
        embedded = self.dropout(embedded).to(device = 'cuda')

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1).to(device = 'cuda')
        
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0)).to(device = 'cuda')

        output = torch.cat((embedded[0], attn_applied[0]), 1).to(device = 'cuda')
        output = self.attn_combine(output).unsqueeze(0).to(device = 'cuda')

        output = F.relu(output).to(device = 'cuda')
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# read the textual data from the source dataset and its corresponding target dataset and pass it on the appropriate
# functions to pre-process it
def prepareData(text, summary):
    data_pairs = [[text[i], summary[i]] for i in range(len(text))]
    data_input = Lang("source",text)
    data_output = Lang("target",summary)

    return data_input, data_output, data_pairs


# generate the indices of the words in a given sentences from the word dictionary of summaries or full-texts
def generate_word_index_from_sentence(lang, sentence):
    words_in_sentence = sentence.split()
    lang_word_index_dictionary = lang.word2idx
    generated_indices = []
    for word in words_in_sentence:
        generated_indices.append(lang_word_index_dictionary[word])

    return generated_indices


# generate a tensor out of the sentence that is given as a parameter
def generate_tensor_from_sentence(lang, sentence):
    indices = generate_word_index_from_sentence(lang, sentence)
    indices.append(lang.EOS_token)

    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1).to(device = 'cuda')

#generate a tensor containing the indices of all of the words in a given list
def generate_tensor_from_all_words(lang, word_list):
    generated_indices = []
    lang_word_index_dictionary = land.word2idx
    for word in word_list:
        generated_indices.append(lang_word_index_dictionary[word])

    #return a tensor made up of the indicies of the provided words
    return torch.tensor(generated_indices, dtype=torch.long, device=device).view(-1,1)
    

# generate tensor from a text input and its corresponding summary pair


def generate_tensors_from_pair(input_text_lang, summary_text_lang, text_summary_pair):
    input_text_tensor = generate_tensor_from_sentence(
        input_text_lang, text_summary_pair[0]).to(device = 'cuda')
    summary_text_tensor = generate_tensor_from_sentence(
        summary_text_lang, text_summary_pair[1]).to(device = 'cuda')

    return(input_text_tensor, summary_text_tensor)


# train the encoder and the decoder models

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=600):
    # initialize the hidden dimension of the encoder
    encoder_hidden_state = encoder.initHidden().to(device = 'cuda')
    # clear the gradients of the encoder optimizer
    encoder_optimizer.zero_grad()
    # clear the gradients of the decoder optimizer
    decoder_optimizer.zero_grad()
    # get the number of tensors of the input text
    input_length = input_tensor.size(0)
    # get the number of tensors of the target text
    target_length = target_tensor.size(0)
    # create a tensor to house the hidden states of each gru cell in the encoder network
    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device).to(device = 'cuda')

    # initilise the value of the loss
    loss = 0

    # iterate over the tensors in the input and push them through the encoder
    for i in range(input_length):
        # push the input tensor and the initial hidden layer into the encoder network
        encoder_output, encoder_hidden_state = encoder(
            input_tensor[i], encoder_hidden_state)
        # collect the outputs of the encoder network for the given input tensor
        encoder_outputs[i] = encoder_output[0, 0]

    # initialize the input of the decoder network with the SOS token
    decoder_input = torch.tensor([[0]], device=device).to(device = 'cuda')

    # initialize the first hidden state of the decoder network
    decoder_hidden_state = encoder_hidden_state

    # iterate through the tensors representing the target texts and push them through the decoder
    # network whiles applying 'teacher forcing' to enable an improvement in performance
    for j in range(target_length):
        decoder_output, decoder_hidden_state, decoder_attention_weights = decoder(
            decoder_input, decoder_hidden_state, encoder_outputs)
        # compute the loss between the target value and the decoder output value
        loss = loss + criterion(decoder_output, target_tensor[j])
        # apply the concept of teacher forcing to appy the correct input to the next gru cell in the next iteration
        decoder_input = target_tensor[j]

    # computes the gradients for all the tensors during the process of back-propagation
    loss.backward()

    # update the weights of the networks in the encoder network
    encoder_optimizer.step()

    # update the weights of the networks in the decoder network
    decoder_optimizer.step()

    # return the average loss across all of the elements in the target sequence
    return loss.item()/target_length


# run the training model
def run_train_model(encoder, decoder, num_iterations, input_lang_def, target_lang_def, sentence_pairs, learning_rate=0.01):
    # training may take some time so print message to indicate that training is taking place
    print("Training is taking place...")
    # keep track of the total losses generated
    total_loss = 0

    # define the encoder optimizer
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # define the decoder optimizer
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # obtain the sentence pairs for training the model
    training_data = [generate_tensors_from_pair(input_lang_def, target_lang_def, random.choice(
        sentence_pairs)) for i in range(num_iterations)]

    # define the criterion to use
    criterion = nn.NLLLoss().to(device='cuda')

    computed_loss = 0

    # iterate through the selected training data pairs and train the data
    for i in range(1, num_iterations+1):
        if i % 10 == 0:
            print('Epoch:{}/{}..............'.format(i, num_iterations), end=' ')
            # print the loss for display
            print("Loss: {:.4f}".format(computed_loss))

        # select a training pair from the training data
        training_data_sentence_pair = training_data[i-1]

        # separate the training data sentence pair into input and target pairs
        input_tensor = training_data_sentence_pair[0].to(device='cuda')
        target_tensor = training_data_sentence_pair[1].to(device='cuda')

        # call the train function on the data and obtain the loss
        computed_loss = train(input_tensor, target_tensor, encoder,
                              decoder, encoder_optimizer, decoder_optimizer, criterion)

        

        # accumulate the loss
        total_loss += computed_loss

    return


# perform an inference on live data points
def inference(encoder, decoder, input_lang_def, output_lang_def, sentence, max_length=600):
    # since inference is being done there is no need to update gradient values
    with torch.no_grad():

        # generate an input tensor from the given sentence
        input_tensor = generate_tensor_from_sentence(input_lang_def, sentence).to(device='cuda')
        input_length = input_tensor.size()[0]

        # initialize the hidden state of the encoder network
        encoder_hidden_state = encoder.initHidden().to(device='cuda')
        # initialize encoder_outputs to keep track of the outputs generated by the encoder
        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device).to(device = 'cuda')


        # for every word representation in the input tensor, push it through the encoder network
        for i in range(input_length):
            encoder_output, encoder_hidden_state = encoder(
                input_tensor[i], encoder_hidden_state)
            encoder_outputs[i] = encoder_output[0, 0]

        # initialize the decoder network

        # as the first input, place in the 'start sentence' token
        decoder_input = torch.tensor([[0]], device=device).to(device = 'cuda')
        # make the first hidden state of the decoder the final hidden state of the encoder
        decoder_hidden_state = encoder_hidden_state

        # keep track of the the decoded words generated by the decoder network
        generated_decoded_words = []

        # initialize the attention weights of the decoder network
        decoder_attention_weights = torch.zeros(max_length, max_length).to(device = 'cuda')

        # iterate through the decoder network to get the decoded words
        for i in range(max_length):
            decoder_output, decoder_hidden_state, decoder_attention_weights_for_current_state = decoder(
                decoder_input, decoder_hidden_state, encoder_outputs)

            decoder_attention_weights[i] = decoder_attention_weights_for_current_state.data
            # return the largest one element from the decoder output
            output_tensor_object, output_long_tensor_object = decoder_output.data.topk(
                1)
            # check if the EOS token was predicted by the model
            if output_long_tensor_object.item() == output_lang_def.EOS_token:
                generated_decoded_words.append('<EOS>')
                break
            else:
                # if the EOS tag was not generated, find out the word that was generated and append it to the generated decoded words
                generated_decoded_words.append(
                    output_lang_def.idx2word[output_long_tensor_object.item()])

            decoder_input = output_long_tensor_object.squeeze().detach()

        return generated_decoded_words, decoder_attention_weights[:max_length]


# test out the prediction quality
def sample(encoder, decoder, input_lang_def, output_lang_def, sentence_pairs, num_iterations, outputfile):
    # open the output file
    outputfile = open(outputfile, 'a')
    precision = 0
    fmeasure = 0
    recall = 0
    for i in range(num_iterations):
        # randomly select the sentence pair
        selected_sentence_pair = random.choice(sentence_pairs)
        # take out the input sentence in the sentence pair
        input_of_selected_sentence_pair = selected_sentence_pair[0]
        # perform an inference on the chosen sentence pair
        generated_words, generated_attention_weights = inference(
            encoder, decoder, input_lang_def, output_lang_def, input_of_selected_sentence_pair)
        # join the list of generated words to form a sentence
        generated_sentence_from_generated_words = ' '.join(generated_words)

        outputfile.write('Original text: '+input_of_selected_sentence_pair+'\n'+' Predicted summary: '+generated_sentence_from_generated_words + '\n'+
                         'Original summary: '+selected_sentence_pair[1]+'\n')
        
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        score = scorer.score(selected_sentence_pair[1], generated_sentence_from_generated_words)

        #print out the rouge score of the predicted sequence
        precision += score['rouge1'].precision
        recall += score['rouge1'].recall
        fmeasure += score['rouge1'].fmeasure
    
    average_precision = precision/num_iterations
    average_recall = recall/num_iterations
    average_fmeasure = fmeasure/num_iterations

    print('average precision: ', average_precision)
    print('average recall: ', average_recall)
    print('average_fmeasure: ', average_fmeasure)
        



    # close the output file
    outputfile.close()

    return (average_precision, average_recall, average_fmeasure)


if __name__ == '__main__':
    total_precision = []
    total_recall = []
    total_fmeasure = []

    # loading and preprocessing the data
    print('PREPROCESSING DATA ...')

    print('Reading data files ...')
 
    extractor('train.json')

    print('Reading stop words file ...')
    
    nltk_english_stop_words = nltk_stop_words('english_stop_words')
    print('preprocessing the source text data ...')
    X = run_preprocess(dataset, nltk_english_stop_words)
    print('preprocessing the target text data ...')
    Y = run_preprocess(target, nltk_english_stop_words)
    print('reorganizing data into frames ...')
    text_dataframe = trim_data(X, Y)
    data_input, data_output, data_pairs = prepareData(
        list(text_dataframe['source_data']), list(text_dataframe['summary']))

    # TRAINING THE MODEL

    print('creating a representation for the source text data ...')
    # get a representation of the input text data so that you can obtain its properties
    input_text_representation = data_input

    #generate a tensor containing the indicies of all words in the source data
    #input_text_tensor = generate_tensor_from_all_words(data_input, data_input.combined_list)

    # get the number of words from the input text data representation as this would represent the input size
    input_size = input_text_representation.total_num_words
    # define the size of the hidden layer (choose an arbitrary value)
    hidden_size = 300

    print('creating an encoder instance ...')
    # Create an instance of the encoder
    encoder = Encoder(input_size, hidden_size)


    print('creating a representation of the target text data ...')
    # get a representation for the output text data
    output_text_representation = data_output
    # get the number of words from the output text data representation as this would represent the output size
    output_size = output_text_representation.total_num_words

    print('creating a decoder instance ...')
    # create an instance of the decoder
    decoder = Decoder(hidden_size, output_size)

    print('TRAINING THE MODEL ...')
    # run the train function

    #train on different sizes of data

    run_train_model(encoder, decoder, 300, input_text_representation,
                    output_text_representation, data_pairs)
    
    print('TESTING THE MODEL ...')
    # test the model to see the level of quality of the generated output

    output = sample(encoder, decoder, input_text_representation,
           output_text_representation, data_pairs, 100, 'results.txt')
    
    total_precision.append(output[0])
    total_recall.append(output[1])
    total_fmeasure.append(output[2])

    

    iteration_count_total = 0
    iteration_count = 50
    initial_value = 350

    while iteration_count_total <= 250:
      print('TRAINING THE MODEL ...')
      run_train_model(encoder, decoder, initial_value, input_text_representation,
                      output_text_representation, data_pairs)
      

      print('TESTING THE MODEL ...')
      output = sample(encoder, decoder, input_text_representation,
           output_text_representation, data_pairs, 100, 'results.txt')
      
      initial_value += iteration_count
      iteration_count_total += iteration_count
      
      total_precision.append(output[0])
      total_recall.append(output[1])
      total_fmeasure.append(output[2])
      

    #plot the graph
    x_axes = [300, 350, 400, 450, 500, 550]

    plt.plot(x_axes, total_precision, label="precision")
    plt.plot(x_axes, total_recall, label="recall")
    plt.plot(x_axes, total_fmeasure, label="fmeasure")

    # naming the x axis
    plt.xlabel('training size')
    # naming the y axis
    plt.ylabel('performance')

    plt.legend()

    plt.title("Performance of model as training size increases")
    


