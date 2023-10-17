import random

## commandline
import argparse

## filepath ops
import os, sys

## nltk
import nltk
#nltk.download()

import re
import math
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline, padded_everygrams
from nltk.lm import MLE, Vocabulary
from nltk.lm.models import InterpolatedLanguageModel, Lidstone, KneserNeyInterpolated, Laplace, StupidBackoff, WittenBellInterpolated
from nltk.util import bigrams, trigrams, ngrams
from nltk import tokenize

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
    textfilter1 = tokenize.sent_tokenize(text.lower())
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
    ngram, vocab = padded_everygram_pipeline(n, train[file_name])
    model_name_up = model_name.upper()
    if model_name_up == "MLE":
        model = MLE(n)

    elif model_name_up == "SB":
        model = StupidBackoff(order=n, alpha=0.9)

    elif model_name_up == "KNI":
        model = KneserNeyInterpolated(order=n, discount=0.75)

    elif model_name_up == "LIDSTONE":
        model = Lidstone(order=n, gamma=0.1)

    elif model_name_up == "WBI":
        model = WittenBellInterpolated(order=n)

    else:
        raise Exception("wrong model name")

    model.fit(ngram, vocab)
    return model

## ngram size
ngram_n = 2
print(f"ngram_n: {ngram_n}")

print("training LMs...(this may take a while) ")
models = {}
for file_name in train.keys():
    ## args.test has already been filtered
    models[file_name] = train_model("LIDSTONE", ngram_n, file_name, train)


def development(train, test):
    for file_name in test.keys():
        correct, total = 0, 0

        for sent in test[file_name]:
            test_ngrams = list(ngrams(pad_both_ends(sent, n=ngram_n), n=ngram_n))

            perplexities = {}
            for file_name_train in train.keys():
                perplexities[file_name_train] = models[file_name_train].perplexity(test_ngrams)
            #print(perplexities)

            pred_file_name = min(perplexities, key=perplexities.get)

            pred_author = author_name(pred_file_name)
            actual_author = author_name(file_name)

            if pred_author == actual_author:
                correct += 1
            total += 1
        
        accuracy = correct / total
        print(author_name(file_name) + f"\t{accuracy*100:.1f}%" + " correct")

    ####--------------#########---------########

def test_file(train, test):

    for sent in test[args.test]:
        test_ngrams = list(ngrams(pad_both_ends(sent, n=ngram_n), n=ngram_n))
        
        perplexities = {}
        for file_name in train.keys():
            perplexities[file_name] = models[file_name].perplexity(test_ngrams)
        # print(perplexities)
        pred_file_name = min(perplexities, key=perplexities.get)
        pred_author = author_name(pred_file_name) # .capitalize()
        print(f"{sent}: {pred_author}", end="\n")
        #print(f"perplexity: {min(perplexities.values())}")


ModelEntry_PostTraining = True
if ModelEntry_PostTraining:
    if args.test:
        test_file(train, test)
    else:
        development(train, test)
        # for i in range(10,35, 5):
        #     Lidstone_models = {}
        #     Lidstone_models = models[file_name] = train_model("LIDSTONE", ngram_n, file_name, train)
        #     print(f"Austen: {Lidstone_models['austen'].generate(10, random_seed=i)} ")
        #     StupidBackoff_models = {}
        #     StupidBackoff_models = train_model("SB", ngram_n, file_name, train)
        #     print(f"Dickens: {StupidBackoff_models['dickens'].generate(10, random_seed=i)}")
        #     print(f"Tolstoy: {StupidBackoff_models['tolstoy'].generate(10, random_seed=i)}")
        #     MLE_models = {}
        #     MLE_models = train_model("MLE", ngram_n, file_name, train)
        #     print(f"Wilde: {MLE_models['wilde'].generate(10, random_seed=i)}")
else:
    print("Nothing")


####---------------###############--------------############
models1 = {}
models2 = {}
models3 = {}
models4 = {}

file_name1 = 'austen.txt'
file_name2 = 'dickens.txt'
file_name3 = 'tolstoy.txt'
file_name4 = 'wilde.txt'

print("Text generation for Austin:-----------------------------------------")
#Text generation for Austin:
num_samples = 5
for random_seed in range(num_samples):
    models1[file_name1] = train_model("LIDSTONE", ngram_n, file_name1, train)
    Lidstone_models = models1[file_name1]
    generated_text = Lidstone_models.generate(10, random_seed=random_seed)
    test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
    ppxt = models[file_name1].perplexity(test_ngrams_p)
    print(f"{author_name(file_name1)}:: model is Lidstone:: Sample-{random_seed+1} :: {generated_text}:: Perplexity score: {ppxt}")

# for random_seed in range(num_samples):
#     models2[file_name1] = train_model("MLE", ngram_n, file_name1, train)
#     MLE_models = models2[file_name1]
#     generated_text = MLE_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name1].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name1)}:: model is MLE:: {generated_text}:: Perplexity score: {ppxt}")

# for random_seed in range(num_samples):
#     models3[file_name1] = train_model("SB", ngram_n, file_name1, train)
#     SB_models = models3[file_name1]
#     generated_text = SB_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name1].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name1)}:: model is StupidBackoff:: {generated_text}:: Perplexity score: {ppxt}")

# for random_seed in range(num_samples):
#     models4[file_name1] = train_model("WBI", ngram_n, file_name1, train)
#     WBI_models = models4[file_name1]
#     generated_text = WBI_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name1].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name1)}:: model is WittenBellInterpolated:: {generated_text}:: Perplexity score: {ppxt}")

print("Text generation for Dickens:-----------------------------------------")
#Text generation for Dickens:
# for random_seed in range(num_samples):
#     models1[file_name2] = train_model("LIDSTONE", ngram_n, file_name2, train)
#     Lidstone_models = models1[file_name2]
#     generated_text = Lidstone_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name2].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name2)}:: model is Lidstone:: {generated_text}:: Perplexity score: {ppxt}")

for random_seed in range(num_samples):
    models2[file_name2] = train_model("MLE", ngram_n, file_name2, train)
    MLE_models = models2[file_name2]
    generated_text = MLE_models.generate(10, random_seed=random_seed)
    test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
    ppxt = models[file_name2].perplexity(test_ngrams_p)
    print(f"{author_name(file_name2)}:: model is MLE::  Sample-{random_seed+1} ::{generated_text}:: Perplexity score: {ppxt}")

# for random_seed in range(num_samples):
#     models3[file_name2] = train_model("SB", ngram_n, file_name2, train)
#     SB_models = models3[file_name2]
#     generated_text = SB_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name2].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name2)}:: model is StupidBackoff:: {generated_text}:: Perplexity score: {ppxt}")

# for random_seed in range(num_samples):
#     models4[file_name2] = train_model("WBI", ngram_n, file_name2, train)
#     WBI_models = models4[file_name2]
#     generated_text = WBI_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name2].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name2)}:: model is WittenBellInterpolated:: {generated_text}:: Perplexity score: {ppxt}")

print("Text generation for Tolstoy:-----------------------------------------")
#Text generation for Tolstoy:
# for random_seed in range(num_samples):
#     models1[file_name3] = train_model("LIDSTONE", ngram_n, file_name3, train)
#     Lidstone_models = models1[file_name3]
#     generated_text = Lidstone_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name3].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name3)}:: model is Lidstone:: {generated_text}:: Perplexity score: {ppxt}")

# for random_seed in range(num_samples):
#     models2[file_name3] = train_model("MLE", ngram_n, file_name3, train)
#     MLE_models = models2[file_name3]
#     generated_text = MLE_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name3].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name3)}:: model is MLE:: {generated_text}:: Perplexity score: {ppxt}")

for random_seed in range(num_samples):
    models3[file_name3] = train_model("SB", ngram_n, file_name3, train)
    SB_models = models3[file_name3]
    generated_text = SB_models.generate(10, random_seed=random_seed)
    test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
    ppxt = models[file_name3].perplexity(test_ngrams_p)
    print(f"{author_name(file_name3)}:: model is StupidBackoff:: Sample-{random_seed+1} ::{generated_text}:: Perplexity score: {ppxt}")

# for random_seed in range(num_samples):
#     models4[file_name3] = train_model("WBI", ngram_n, file_name3, train)
#     WBI_models = models4[file_name3]
#     generated_text = WBI_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name3].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name3)}:: model is WittenBellInterpolated:: {generated_text}:: Perplexity score: {ppxt}")

print("Text generation for Wilde:-----------------------------------------")
#Text generation for Wilde:
# for random_seed in range(num_samples):
#     models1[file_name4] = train_model("LIDSTONE", ngram_n, file_name4, train)
#     Lidstone_models = models1[file_name4]
#     generated_text = Lidstone_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name4].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name4)}:: model is Lidstone:: {generated_text}:: Perplexity score: {ppxt}")

# for random_seed in range(num_samples):
#     models2[file_name4] = train_model("MLE", ngram_n, file_name4, train)
#     MLE_models = models2[file_name4]
#     generated_text = MLE_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name4].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name4)}:: model is MLE:: {generated_text}:: Perplexity score: {ppxt}")

# for random_seed in range(num_samples):
#     models3[file_name4] = train_model("SB", ngram_n, file_name4, train)
#     SB_models = models3[file_name4]
#     generated_text = SB_models.generate(10, random_seed=random_seed)
#     test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
#     ppxt = models[file_name4].perplexity(test_ngrams_p)
#     print(f"{author_name(file_name4)}:: model is StupidBackoff:: {generated_text}:: Perplexity score: {ppxt}")

for random_seed in range(num_samples):
    models4[file_name4] = train_model("WBI", ngram_n, file_name4, train)
    WBI_models = models4[file_name4]
    generated_text = WBI_models.generate(10, random_seed=random_seed)
    test_ngrams_p = list(ngrams(pad_both_ends(generated_text, n=ngram_n), n=ngram_n))
    ppxt = models[file_name4].perplexity(test_ngrams_p)
    print(f"{author_name(file_name4)}:: model is WittenBellInterpolated::  Sample-{random_seed+1} :: {generated_text}:: Perplexity score: {ppxt}")
