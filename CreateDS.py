import numpy as np
import torch
import torchaudio
import os
from torch.utils.data import Dataset
import pandas as pd
torchaudio.set_audio_backend("soundfile")

def congifure_dict(number_of_epochs=6,batch_size=8):
    configuration_dict = {'number_of_epochs': number_of_epochs, 'batch_size': batch_size,
                          'dropout': 0.25, 'base_lr': 0.005,
                          'number_of_mel_filters': 64, 'resample_freq': 22050}
    return configuration_dict

class SoundDataset(Dataset):
    def __init__(self, csv_path, file_path, resample_freq=0, return_audio=False):
        self.file_path = file_path
        self.file_names = []
        self.labels = []
        self.n_mels = 64
        self.return_audio = return_audio
        self.resample = resample_freq

        csvData = pd.read_csv(csv_path)
        csvData=csvData.loc[:, ~csvData.columns.str.match("Unnamed")]
        fi=list(csvData.columns).index('path')
        li=list(csvData.columns).index('classID')
        for i in range(0, len(csvData)):
            self.file_names.append(csvData.iloc[i, fi])
            self.labels.append(csvData.iloc[i, li])

    def __getitem__(self, index):
        path = os.path.join(self.file_path, self.file_names[index])
        soundData, sample_rate = torchaudio.load(path,normalize=True)

        if self.resample > 0:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample)
            soundData = resample_transform(soundData)

        soundData = torch.mean(soundData, dim=0, keepdim=True)
        
        melspectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.resample, n_mels=self.n_mels)
        melspectrogram = melspectrogram_transform(soundData)
        melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)

        fixed_length = 3 * (self.resample // 200)
        if melspectogram_db.shape[2] < fixed_length:
            melspectogram_db = torch.nn.functional.pad(melspectogram_db, (0, fixed_length - melspectogram_db.shape[2]))
        else:
            melspectogram_db = melspectogram_db[:, :, :fixed_length]

        if self.return_audio:
            fixed_length = 3 * self.resample
            if soundData.numel() < fixed_length:
                soundData = torch.nn.functional.pad(soundData, (0, fixed_length - soundData.numel())).numpy()
            else:
                soundData = soundData[0, :fixed_length].reshape(1, fixed_length).numpy()
        else:
            soundData = np.array([])

        return soundData, self.resample, melspectogram_db, self.labels[index]

    def __len__(self):
        return len(self.file_names)

def create_train(csv_path,audio_path,batch_size,return_a):
    set = SoundDataset(csv_path, audio_path ,
                                  resample_freq=22050, return_audio=return_a)
    print("Размер набора: " + str(len(set)))
    loader = torch.utils.data.DataLoader(set, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    return loader
