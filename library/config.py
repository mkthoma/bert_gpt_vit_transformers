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
            "training_path": "/content/bert_gpt_vit_transformers/library/data/bert/training.txt",
            "vocab_path": "/content/bert_gpt_vit_transformers/library/data/bert/vocab.txt"

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
                "dataset_path": "/content/bert_gpt_vit_transformers/library/data/gpt/english.txt",
                "NUM_HEAD": num_head,
                "NUM_EMBED": num_embed,
                "NUM_LAYER": 6,
                "DROPOUT": 0.2}
    return config



def vit_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = "/content/bert_gpt_vit_transformers/library/data/vit"
    train_dir = image_path + "/train"
    test_dir = image_path + "/test"

    config = {"device": device,
            "train_dir": train_dir,
            "test_dir": test_dir,
            "IMG_SIZE": 224,
            "BATCH_SIZE": 32,
            "patch_size": 16,
            "img_size": 224, # Training resolution from Table 3 in ViT paper
            "in_channels": 3, # Number of channels in input image
            "num_transformer_layers": 12, # Layers from Table 1 for ViT-Base
            "embedding_dim": 768, # Hidden size D from Table 1 for ViT-Base
            "mlp_size": 3072, # MLP size from Table 1 for ViT-Base
            "num_heads": 12, # Heads from Table 1 for ViT-Base
            "attn_dropout": 0, # Dropout for attention projection
            "mlp_dropout": 0.1, # Dropout for dense/MLP layers 
            "embedding_dropout": 0.1,} # Dropout for patch and position embeddings
    return config

