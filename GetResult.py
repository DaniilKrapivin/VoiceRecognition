from NetConf import create_model
import pandas as pd
import torch
import os
import numpy as np
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def prediction(param_pth,audio_pth,n_classes,n_epochs,useD=True):
    model, _, _, _ = create_model(n_classes, n_epochs)
    model.eval()
    model.load_state_dict(torch.load(param_pth))
    wav, sr = torchaudio.load(audio_pth, normalize=True)
    resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
    soundData = resample_transform(wav)
    soundData = torch.mean(soundData, dim=0, keepdim=True)
    melspectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=64)
    melspectrogram = melspectrogram_transform(soundData)
    melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)
    fixed_length = 3 * (22050 // 200)
    if melspectogram_db.shape[2] < fixed_length:
        inputs = torch.nn.functional.pad(melspectogram_db, (0, fixed_length - melspectogram_db.shape[2]))
    else:
        inputs = melspectogram_db[:, :, :fixed_length]
    if useD:
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        inputs=inputs.to(device)
        model=model.to(device)
    outputs = model(inputs.unsqueeze(dim=0))
    _, predicted = torch.max(outputs, 1)
    return int(predicted[0])
