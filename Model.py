import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torchaudio

class Model(nn.Module):
    def __init__(self, text_vocab_size,
                 embedding_dim, d_model, n_head, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout=0.1,
                 ff_dim=2048, max_len=512, activation="relu",
                 embedding_dropout=0.1,
                 layer_norm_eps=1e-12, use_position_encoding=True,
                 position_encoding_dropout=0.1,
                 n_mels=128, n_fft=2048, hop_length=512,
                 sample_rate=22050):
        super(Model, self).__init__()

        # ----------- Hyperparameters -----------
        self.text_vocab_size = text_vocab_size  # Size of text-vocabulary
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.activation = activation
        self.embedding_dropout = embedding_dropout
        self.layer_norm_eps = layer_norm_eps
        self.position_encoding_dropout = position_encoding_dropout
        self.use_position_encoding = use_position_encoding

        # Mel-spectrogram parameters
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # ----------- Embeddings ------------
        self.text_embedding = nn.Embedding(self.text_vocab_size, self.embedding_dim)
        self.music_embedding = nn.Linear(self.n_mels, self.embedding_dim)  # Используем линейный слой для встраивания мел-спектрограммы

        # ------------ Encoder ------------
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # ----------- Decoder -----------
        decoder_layers = nn.TransformerDecoderLayer(d_model, n_head, ff_dim, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)

        self.output_layer = nn.Linear(d_model, self.n_mels)  # Выходной слой для мел-спектрограммы

        # Мел-спектрограммный преобразователь
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

    def forward(self, text: str, music):
        """
        :param text: just a string used to give a prompt to model
        :param music: music input what can be used to generate long music step by step, or just random seed
        :return: output of the model, music
        """

        # Встраивание текста
        text_embedded = self.text_embedding(text)

        # Встраивание музыки (мел-спектрограммы)
        music_embedded = self.music_embedding(music)
        # Встраивание музыки (мел-спектрограммы)
        # music_embedded = self.music_embedding(music.view(music.size(0), -1))
        print(music.shape)
        # music_embedded = self.music_embedding(music.view(music.size(0), self.n_mels, -1))
        # music_embedded = self.music_embedding(music.view(music.size(0), -1))
        text_encoded = self.encoder(text_embedded)
        music_decoded = self.decoder(tgt=music_embedded, memory=text_encoded)
        music_output = self.output_layer(music_decoded)
        music_output = music_output / torch.max(music_output)

        return music_output

    def generate_music(self, text, initial_music=None, steps=10):
        generated_music = []
        # if initial_music is None:
        #     initial_music = torch.randn(1, self.n_mels, 1)
        if initial_music is None:
            initial_music = torch.randn(1, self.n_mels * 1)

        current_music = initial_music

        for _ in range(steps):
            music_output = self.forward(text, current_music)
            generated_music.append(music_output)
            current_music = music_output

        return generated_music
    def save(self, path):
        '''
        :param path:
        :return:
        '''
        torch.save(self.state_dict(), path)