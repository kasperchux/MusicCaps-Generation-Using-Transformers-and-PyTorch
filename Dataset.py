import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import torchaudio
import torch.nn.functional as F
import os

class MusicDataset(Dataset):
    def __init__(self, dataset_name="google/MusicCaps",tokenizer_name="openai-community/gpt2", n_mels=128,
                 n_fft=2048, hop_length=512, sample_rate=22050):
        '''
        :param text_data: prompt to model
        :param music_data: music, output from model
        '''
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.dataset = load_dataset(dataset_name)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        self.remove_missing_audio()
        self.get_max_len()

    def __len__(self) -> int:
        '''
        :return: length of the dataset
        '''
        return len(self.dataset)

    def get_max_len(self):
        self.max_audio_length = 958728 # calculated first time
        # for item in self.dataset['train']:
        #     ytid = item['ytid']
        #     audio_file_path = f"./music_data/{ytid}.wav"
        #     waveform, sample_rate = torchaudio.load(audio_file_path)
        #     self.max_audio_length = max(self.max_audio_length, waveform.shape[1])
        # print(f"{self.max_audio_length} -----------------------")

    def remove_missing_audio(self):
        idx_to_remove = []

        for idx, item in enumerate(self.dataset['train']):
            ytid = item['ytid']
            audio_file_path = f"./music_data/{ytid}.wav"
            if not os.path.isfile(audio_file_path):
                idx_to_remove.append(idx)

        self.dataset = self.dataset.filter(lambda ex, idx: idx not in idx_to_remove, with_indices=True)

        print(f'Удалено {len(idx_to_remove)} строк из датасета')

    def __getitem__(self, idx: int):
        item = self.dataset['train'][idx]
        ytid = item['ytid']
        caption = item['caption']

        audio_file_path = f"./music_data/{ytid}.wav"
        waveform, sample_rate = torchaudio.load(audio_file_path)

        waveform = F.pad(waveform, (0, self.max_audio_length - waveform.shape[1]), 'constant', 0)

        mel_spectrogram = self.mel_spectrogram(waveform)
        mel_spectrogram = mel_spectrogram.transpose(1, 2)

        text_tokens = self.tokenizer(caption, return_tensors="pt")["input_ids"].squeeze(1)
        print(text_tokens.shape)

        return {
            "prompt": text_tokens,
            "music": mel_spectrogram,
        }


