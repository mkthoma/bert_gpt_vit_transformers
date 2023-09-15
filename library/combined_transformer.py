
from .model import *
from .train import *
from .utils import *
from config import *

def TransformerCombined(name):
    if name == "BERT":
        config=bert_config()
        data_loader, dataset = bert_dataloader(config["training_path"], config["vocab_path"])

        model = bert_transformer(config["n_code"], config["n_heads"], config["embed_size"], config["inner_ff_size"], len(dataset.vocab), config["seq_len"], config["dropout"])
        return model, data_loader, dataset
    
    elif name == "GPT":
        config=gpt_config()
        train_data, val_data, vocab_size = gpt_dataset(config["dataset_path"])
        model = gpt_transformer(vocab_size, config["NUM_EMBED"], config["BLOCK_SIZE"], config["NUM_HEAD"], config["NUM_LAYER"], config["DROPOUT"])
        return model, train_data, val_data


