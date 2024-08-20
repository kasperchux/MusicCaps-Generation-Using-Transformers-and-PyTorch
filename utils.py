import torch
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def train_step(model, dataloader, optimizer, loss_fn, epoch, device, rank, train_losses, epoch_losses):
    '''
    :param model: model, what we will use for training
    :param dataloader: dataloader, what we will use for training
    :param optimizer: optimizer, what we will use for training the model
    :param loss_fn: loss function, what we will use to rate an error of the model
    :param epoch: current epoch
    :param device: device for training
    :param rank: rank of the current process
    :param train_losses: list to store all losses during training
    :param epoch_losses: list to store losses for each epoch
    :return: None
    '''
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} (Rank {rank})")):
        optimizer.zero_grad()
        text_input = batch["text"].to(device)
        music_input = batch["music"].to(device)
        music_output = model(text_input, music_input)
        loss = loss_fn(music_output, music_input)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_losses.append(loss.item())

    if rank == 0:
        epoch_losses.append(total_loss / len(dataloader.dataset))
        print(f"Average loss on train {epoch+1}: {epoch_losses[-1]}")

def evaluate(model, dataloader, loss_fn, device, rank, val_losses, epoch_losses):
    model.eval() # Switch model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            text_input = batch["text"].to(device)
            music_input = batch["music"].to(device)
            music_output = model(text_input, music_input)
            loss = loss_fn(music_output, music_input)
            total_loss += loss.item()
            val_losses.append(loss.item())

    if rank == 0:
        epoch_losses.append(total_loss / len(dataloader.dataset))
        print(f"Agerage loss on validation: {epoch_losses[-1]}")


def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=10, device="cuda"):
    train_losses = []
    val_losses = []
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            text_input = batch["prompt"].to(device)
            music_input = batch["music"].to(device)
            music_output = model(text_input, music_input)
            loss = loss_fn(music_output, music_input)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_losses.append(loss.item())

        epoch_train_losses.append(total_loss / len(train_dataloader.dataset))
        print(f"Средняя потеря на эпохе {epoch+1}: {epoch_train_losses[-1]}")

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                text_input = batch["prompt"].to(device)
                music_input = batch["music"].to(device)
                music_output = model(text_input, music_input)
                loss = loss_fn(music_output, music_input)
                total_loss += loss.item()
                val_losses.append(loss.item())

        epoch_val_losses.append(total_loss / len(val_dataloader.dataset))
        print(f"Средняя потеря на валидации: {epoch_val_losses[-1]}")

    return train_losses, val_losses, epoch_train_losses, epoch_val_losses