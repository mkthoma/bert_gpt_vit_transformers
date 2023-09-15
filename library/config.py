import torch


def bert_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1024
    seq_len = 20
    embed_size = 128
    inner_ff_size = embed_size * 4
    n_heads = 8
    n_code = 8
    n_vocab = 40000
    dropout = 0.1
    config = {"device": device,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "embed_size": embed_size,
            "embed_size": embed_size,
            "inner_ff_size": inner_ff_size,
            "n_heads": n_heads,
            "n_code": n_code,
            "n_vocab": n_vocab,
            "dropout": dropout,
            "training_path": "data/bert/training.txt",
            "vocab_path": "data/bert/vocab.txt"

            }
    return config


def gpt_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_head = 6
    num_embed = num_head * 128
    config = {"BATCH_SIZE": 32,  # how many independent sequences will we process in parallel?
                "BLOCK_SIZE": 64,  # what is the maximum context length for predictions?
                "MAX_ITER": 5000,  # number of training iterations
                "EVAL_INTER": 500,
                "LEARNING_RATE": 3e-4,
                "DEVICE": device,
                "dataset_path": "data/gpt/english.txt",
                "NUM_HEAD": num_head,
                "NUM_EMBED": num_embed,
                "NUM_LAYER": 6,
                "DROPOUT": 0.2}
    return config



def vit_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = "data/vit"
    train_dir = image_path + "/train"
    test_dir = image_path + "/test"
    config = {"device": device,
            "train_dir": train_dir,
            "test_dir": test_dir,
            "IMG_SIZE": 224,
            "BATCH_SIZE": 32,
            "patch_size": 16}
    return config

