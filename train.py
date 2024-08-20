import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Model import Model
from Dataset import MusicDataset
from Config import Config
from utils import train_step, evaluate, train
import matplotlib.pyplot as plt

text_vocab_size = 50257
embedding_dim = 512
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
ff_dim = 2048
max_len = 512
dropout = 0.1
learning_rate = 1e-4
epochs = 1
batch_size = 64

if __name__ == "__main__":
    model = Model(
        text_vocab_size=text_vocab_size,
        embedding_dim=embedding_dim,
        d_model=d_model,
        n_head=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=ff_dim,
        dropout=dropout,
        ff_dim=ff_dim,
        max_len=max_len,
        activation="relu",
        embedding_dropout=0.1,
        layer_norm_eps=1e-12,
        use_position_encoding=True,
        position_encoding_dropout=0.1,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        sample_rate=22050
    )

    dataset = MusicDataset()

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_losses, val_losses, epoch_train_losses, epoch_val_losses = train(
        model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=epochs,device=device
    )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(epoch_train_losses, label="Train Loss (Epoch)")
    plt.plot(epoch_val_losses, label="Validation Loss (Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean loss")
    plt.legend()
    plt.show()

