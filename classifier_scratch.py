import random

## commandline
import argparse

## filepath ops
import os, sys

import re
import math

#Understanding what to do before implementation:
#Step-1: Define help description for input arguments from cmd line and create argument parser to store the cmd line arguments.
#Step-2: Load the text data from the needed files based on cmd line argument.
#Step-3: Preprocess the loaded text data.
#Step-4: Split the data as per the criteria defined as per the arguments from cmd line. 
#Step-5: Build a function for training model
#Step-6: Build a function for development
#Step-7: Build a function for testfile testing


parser = argparse.ArgumentParser(description='Author attribution')
parser.add_argument('authorlist', type=str, help='It is a file containing path to author list file')
parser.add_argument('-test', type=str, help='It is a flag for the path to testfile')

args = parser.parse_args()

## get authorlist from cmd
with open(args.authorlist, 'r') as f:
    author_list = f.read()
file_names = author_list.splitlines()
author_name = lambda file_names : file_names.rsplit(".",1)[0] #Remove the .txt part and just store the author name

## load the text
texts = {}
for file_name in file_names:
  with open("ngram_authorship_train/" + file_name, "r") as f1:
      texts[file_name] = f1.read()

## get testfile from cmd if the cmd argument contains test flag
if args.test:
    with open(args.test, 'r') as f:
        test_file = f.read() #This is where it reads the content in the testfile passed at the cmd line as argument
    texts[args.test] = test_file

def preprocess_text(text):
    # Lower case
    textfilter1 = re.split(r'[.!?]', text.lower())
    # Remove newline
    textfilter2 = [sent.replace('\n', ' ').replace('-', ' ') for sent in textfilter1]
    # Remove punctuations
    textfilter3 = [re.sub(r'[^A-Za-z0-9- ]+', '', sent) for sent in textfilter2]
    # Remove extra space
    textfilter4 = [' '.join(sent.split()) for sent in textfilter3]
    # Split sentence into word lists    
    textfilter5 = [sent.split(' ') for sent in textfilter4]
    ## Remove empty lists
    textfilter6 = [lst for lst in textfilter5 if lst]
    return textfilter6

texts = {file_name:preprocess_text(text) for (file_name, text) in texts.items()}


test, train = {}, {}
if not args.test: #If the test flag is not passed in the cmd line argument, split the data into 90% for
    # training and 10% for development dataset (basically validation dataset).
    def split_train_dataset(lst, ratio=0.1):
        n = len(lst)
        n1 = int(n * ratio / (ratio + 1))
        indices = random.sample(range(n), n)
        lst1 = [lst[i] for i in indices[:n1]]
        lst2 = [lst[i] for i in indices[n1:]]
        return lst1, lst2
    print("splitting into training and development datasets")
    for (file_name, text) in texts.items():
        test[file_name], train[file_name] = split_train_dataset(text)
else: #If test flag is passed then the above if statement would not be executed and the training datset is not split
    for (file_name, text) in texts.items():
        if file_name != args.test:
            train[file_name] = texts[file_name]
        else:
            test[file_name] = texts[file_name]


## MLE, SB, LIDSTONE, KNI models
def train_model(model_name, n, file_name, train):
    ngrams = []
    for sentence in train[file_name]:
        ngrams.extend(zip(*[sentence[i:] for i in range(n)]))
    ngram_counts = dict()
    for ngram in ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    total_ngrams = len(ngrams)  # Calculate total_ngrams

    if model_name.upper() == "MLE":
        model = ngram_counts
    else:
        raise Exception("Wrong model name")
    return model, total_ngrams  # Return total_ngrams

## ngram size
ngram_n = 2

print("training LMs...(this may take a while) ")
models = {}
total_ngrams = 0  # Initialize total_ngrams
for file_name in train.keys():
    model, ngrams = train_model("MLE", ngram_n, file_name, train)  # Get model and total_ngrams
    models[file_name] = model
    total_ngrams += ngrams  # Update total_ngrams

def development(train, test):
    for file_name in test.keys():
        correct, total = 0, 0

        for sentence in test[file_name]:
            ngrams = list(zip(*[sentence[i:] for i in range(ngram_n)]))
            perplexities = {}
            for file_name_train in train.keys():
                model = models[file_name_train]
                perplexity = 0.0
                for ngram in ngrams:
                    ngram_count = model.get(ngram, 0)
                    perplexity += -1.0 / len(ngrams) * math.log((ngram_count + 1) / (total_ngrams + len(model)))
                perplexities[file_name_train] = perplexity

            pred_file_name = min(perplexities, key=perplexities.get)
            pred_author = author_name(pred_file_name)
            actual_author = author_name(file_name)

            if pred_author == actual_author:
                correct += 1
            total += 1

        accuracy = correct / total
        print(author_name(file_name) + f"\t{accuracy * 100:.1f}%" + " correct")

def test_file(train, test):
    for sentence in test[args.test][:-1]:
        ngrams = list(zip(*[sentence[i:] for i in range(ngram_n)]))
        perplexities = {}
        for file_name in train.keys():
            model = models[file_name]
            perplexity = 0.0
            for ngram in ngrams:
                ngram_count = model.get(ngram, 0)
                perplexity += -1.0 / len(ngrams) * math.log((ngram_count + 1) / (total_ngrams + len(model)))
            perplexities[file_name] = perplexity
        pred_file_name = min(perplexities, key=perplexities.get)
        pred_author = author_name(pred_file_name)
        print(f"{sentence}: {pred_author}")


ModelEntry_PostTraining = True
if ModelEntry_PostTraining:
    if args.test:
        test_file(train, test)
    else:
        development(train, test)
else:
    print("Nothing")