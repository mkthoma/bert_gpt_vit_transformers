from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import torch
import torch
from transformers import AutoTokenizer  # pip install transformers
import os
from collections import Counter
from os.path import exists
import torch
import re
from .config import *


################################################################################################################################################
################################################################################################################################################
################################################## DATASET FOR BERT ############################################################################
################################################################################################################################################
################################################################################################################################################

class SentencesDataset(Dataset):
    #Init dataset
    def __init__(self, sentences, vocab, seq_len):
        dataset = self
        
        dataset.sentences = sentences
        dataset.vocab = vocab + ['<ignore>', '<oov>', '<mask>']
        dataset.vocab = {e:i for i, e in enumerate(dataset.vocab)} 
        dataset.rvocab = {v:k for k,v in dataset.vocab.items()}
        dataset.seq_len = seq_len
        
        #special tags
        dataset.IGNORE_IDX = dataset.vocab['<ignore>'] #replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>'] #replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>'] #replacement tag for the masked word prediction task
    
    
    #fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self
        
        #while we don't have enough word to fill the sentence for a batch
        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(index % len(dataset)))
            index += 1
        
        #ensure that the sequence is of length seq_len
        s = s[:dataset.seq_len]
        [s.append(dataset.IGNORE_IDX) for i in range(dataset.seq_len - len(s))] #PAD ok
        
        #apply random mask
        s = [(dataset.MASK_IDX, w) if random.random() < p_random_mask else (w, dataset.IGNORE_IDX) for w in s]
        
        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'target': torch.Tensor([w[1] for w in s]).long()}

    #return length
    def __len__(self):
        return len(self.sentences)

    #get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s] 
        return s
    

# DATALOADER FOR BERT
def bert_dataloader(sentences_path, vocab_path):
    batch_size = bert_config()["batch_size"]
    seq_len = bert_config()["seq_len"]
    n_vocab = bert_config()["n_vocab"]
    sentences = open(sentences_path).read().lower().split('\n')
    special_chars = ',?;.:/*!+-()[]{}"\'&'
    sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    sentences = [[w for w in s if len(w)] for s in sentences]
    if not exists(vocab_path):
        words = [w for s in sentences for w in s]
        vocab = Counter(words).most_common(n_vocab) #keep the N most frequent words
        vocab = [w[0] for w in vocab]
        open(vocab_path, 'w+').write('\n'.join(vocab))
    else:
        vocab = open(vocab_path).read().split('\n')
    dataset = SentencesDataset(sentences, vocab, seq_len)
    kwargs = {'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return data_loader, dataset


################################################################################################################################################
################################################################################################################################################
################################################## DATASET FOR GPT #############################################################################
################################################################################################################################################
################################################################################################################################################

def encode(text_seq: str, tokenizer: any) -> torch.Tensor:
    """
    Function to encode input text using a pre-trained tokenizer and vectorized lookups
    """
    # tokenize the input text
    tokens = tokenizer.tokenize(text_seq)
    # convert the tokens to their corresponding ids
    token_indices = tokenizer.convert_tokens_to_ids(tokens)
    token_indices = torch.tensor(token_indices, dtype=torch.long)
    return token_indices

def gpt_dataset(dataset_path):

    path_do_data = dataset_path
    data_raw = open(path_do_data, encoding="utf-8").read()
    # we use pretrained BERT tokenizer for performance improvements
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    # data_raw = data_raw[4000000:] # short dataset

    # train/val split
    data = encode(text_seq=data_raw, tokenizer=tokenizer)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size


################################################################################################################################################
################################################################################################################################################
################################################## DATASET FOR VIT  ############################################################################
################################################################################################################################################
################################################################################################################################################

NUM_WORKERS = os.cpu_count()-1

def vit_dataloader(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names, train_data, test_data