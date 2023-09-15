import torch
from .model import *
from .train import *
from .utils import *
from .config import *

def TransformerCombined(name, iterations=10):
    if name == "BERT":
        config = bert_config()
        data_loader, dataset = bert_dataloader(config["training_path"], config["vocab_path"])

        model = bert_transformer(config["n_code"], config["n_heads"], config["embed_size"], config["inner_ff_size"], len(dataset.vocab), config["seq_len"], config["dropout"])
        model = model.cuda()
        bert_training(model, data_loader, dataset, iterations=iterations, print_each=1)


    elif name == "GPT":
        config = gpt_config()
        train_data, val_data, vocab_size = gpt_dataset(config["dataset_path"])
        model = gpt_transformer(vocab_size, config["NUM_EMBED"], config["BLOCK_SIZE"], config["NUM_HEAD"], config["NUM_LAYER"], config["DROPOUT"])
        model = model.cuda()
        gpt_training(model, train_data, val_data, config["LEARNING_RATE"], config["BLOCK_SIZE"], config["BATCH_SIZE"], config["DEVICE"], 
                    iterations = iterations, print_each=1)

    elif name == "VIT":
            config = vit_config()
            train_dir = config["train_dir"] 
            test_dir = config["test_dir"]
            IMG_SIZE = config["IMG_SIZE"]
            transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])  
            batch_size = config["BATCH_SIZE"]
            img_size = config["img_size"]
            in_channels = config["in_channels"]
            patch_size = config["patch_size"]
            num_transformer_layers = config["num_transformer_layers"]
            embedding_dim = config["embedding_dim"]
            mlp_size = config["mlp_size"]
            num_heads = config["num_heads"] 
            attn_dropout = config["attn_dropout"]
            mlp_dropout = config["mlp_dropout"]
            embedding_dropout = config["embedding_dropout"] 

            train_dataloader, test_dataloader, class_names, train_data, test_data = vit_dataloader(train_dir, test_dir, transform, batch_size)
            
            model = vit_transformer(img_size, in_channels, patch_size, num_transformer_layers, embedding_dim,mlp_size, num_heads, attn_dropout, mlp_dropout, embedding_dropout, len(train_data.classes))
        
            optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=0.3) 
            loss_fn = torch.nn.CrossEntropyLoss()            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            vit_train(model, train_dataloader, test_dataloader, optimizer, loss_fn, iterations, device)

