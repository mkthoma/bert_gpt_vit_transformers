import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import torch
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


################################################################################################################################################
################################################################################################################################################
################################################## BERT TRAINING ###############################################################################
################################################################################################################################################
################################################################################################################################################

def get_bert_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

def bert_training(model, data_loader, dataset, iterations=10, print_each=1):
    # Optimizer
    optim_kwargs = {'lr':1e-4, 'weight_decay':1e-4, 'betas':(.9,.999)}
    print('initializing optimizer and loss...')
    optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    loss_model = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)

    # Train
    print('training...')
    model.train()
    batch_iter = iter(data_loader)
    for it in range(iterations):
        
        #get batch
        batch, batch_iter = get_bert_batch(data_loader, batch_iter)
        
        #infer
        masked_input = batch['input']
        masked_target = batch['target']
        
        masked_input = masked_input.cuda(non_blocking=True)
        masked_target = masked_target.cuda(non_blocking=True)
        output = model(masked_input)
        
        #compute the cross entropy loss 
        output_v = output.view(-1,output.shape[-1])
        target_v = masked_target.view(-1,1).squeeze()
        loss = loss_model(output_v, target_v)
        
        #compute gradients
        loss.backward()
        
        #apply gradients
        optimizer.step()
        
        #print step
        if it % print_each == 0:
            print('ITERATION:', it, 
                ' | Loss', np.round(loss.item(),2),
                ' | ΔW:', round(model.embeddings.weight.grad.abs().sum().item(),3))
        
        #reset gradients
        optimizer.zero_grad()
        

################################################################################################################################################
################################################################################################################################################
################################################## GPT TRAINING ###############################################################################
################################################################################################################################################
################################################################################################################################################

def get_gpt_batch(data: list[str], block_size: int, batch_size: int, DEVICE):
    """
    This is a simple function to create batches of data.
    GPUs allow for parallel processing we can feed multiple chunks at once
    so that's why we would need batches - how many independant sequences
    will we process in parallel.

    Parameters:
    data: list[str]: data to take batch from
    block_size (int): size of the text that is proccessed at once
    batch_size (int): number of sequences to process in parallel

    Returns:
    x, y: a tuple with token sequence and token target
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # we stack batch_size rows of sentences
    # so x and y are the matrices with rows_num=batch_size
    # and col_num=block_size
    x = torch.stack([data[i : i + block_size] for i in ix])
    # y is x shifted one position right - because we predict
    # word in y having all the previous words as context
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(
    data: list[str],
    model: torch.nn.Module,
    block_size: int,
    batch_size: int,
    eval_iters: int = 10,
):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_gpt_batch(data=data, block_size=block_size, batch_size=batch_size)
        logits, loss = model.forward(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

def gpt_training(model, train_data, val_data, LEARNING_RATE, BLOCK_SIZE, BATCH_SIZE, DEVICE, MAX_ITER = 10, print_each=1):

    # optimizer takes the model's parameters and the learning rate as input,
    # and updates the parameters during the training process in order to
    # minimize the loss function.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(MAX_ITER):

        # every print_each evaluate the loss on train and val sets
        if step % print_each == 0 or step == MAX_ITER - 1:
            loss_train = estimate_loss(
                data=train_data, model=model, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
            )
            loss_val = estimate_loss(
                data=val_data, model=model, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
            )
            print("Step {:10} | Train Loss {:6.4f} | Validation Loss {:6.4f}".format(step, loss_train, loss_val))

        # sample a batch of data
        xb, yb = get_gpt_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, DEVICE=DEVICE)
        logits, loss = model.forward(xb, yb)
        # zero_grad() method sets the gradients of all parameters in the optimizer to zero
        optimizer.zero_grad(set_to_none=True)
        # backward() method on the loss variable calculates the gradients 
        # of the loss with respect to the model's parameters.
        loss.backward()
        # step() method on the optimizer updates the model's parameters 
        # using the calculated gradients, in order to minimize the loss.
        optimizer.step()


################################################################################################################################################
################################################################################################################################################
################################################## ViT TRAINING ###############################################################################
################################################################################################################################################
################################################################################################################################################


def vit_train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               clip_norm: bool=False,
               lr_schedule=None,
               ) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # Clipnorm
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def vit_test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def vit_train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          clip_norm: bool=False,) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through vit_train_step() and vit_test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = vit_train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device,
                                          clip_norm= clip_norm)
        test_loss, test_acc = vit_test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

