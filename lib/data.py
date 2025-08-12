# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import os
from datasets import load_dataset, load_from_disk

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):

    traindir = "data/traindata_wikitext2.hf"
    if os.path.exists(traindir):
        traindata = load_from_disk(traindir)
    else:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        traindata.save_to_disk(traindir)

    testdir = "data/testdata_wikitext2.hf"
    if os.path.exists(testdir):
        testdata = load_from_disk(testdir)
    else:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testdata.save_to_disk(testdir)

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process wikitext2 dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):

    traindir = "data/traindata_ptb.hf"
    if os.path.exists(traindir):
        traindata = load_from_disk(traindir)
    else:
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        traindata.save_to_disk(traindir)

    testdir = "data/testdata_ptb.hf"
    if os.path.exists(testdir):
        testdata = load_from_disk(testdir)
    else:
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
        testdata.save_to_disk(testdir)

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['sentence']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):

    traindir = "data/traindata_c4.hf"
    if os.path.exists(traindir):
        traindata = load_from_disk(traindir)
    else:
        traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
                             keep_in_memory=True, ignore_verifications=True)
        traindata.save_to_disk(traindir)

    valdir = "data/valdata_c4.hf"
    if os.path.exists(valdir):
        valdata = load_from_disk(valdir)
    else:
        valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation', keep_in_memory=True, ignore_verifications=True)
        valdata.save_to_disk(valdir)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_ja(nsamples, seed, seqlen, tokenizer):

    traindir = "data/traindata_ja_c4.hf"
    if os.path.exists(traindir):
        traindata = load_from_disk(traindir)
    else:
        traindata = load_dataset('allenai/c4', data_files={'train': 'multilingual/c4-ja.tfrecord-00000-of-01024.json.gz'}, split='train',
                             keep_in_memory=True, ignore_verifications=True)
        traindata.save_to_disk(traindir)

    valdir = "data/valdata_ja_c4.hf"
    if os.path.exists(valdir):
        valdata = load_from_disk(valdir)
    else:
        valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation', keep_in_memory=True, ignore_verifications=True)
        valdata.save_to_disk(valdir)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ja" in name:
        return get_ja(nsamples, seed, seqlen, tokenizer)
