# VoiceRecognition
#IMPORTS: Необходимые для работы

from CreateDS import SoundDataset, create_train \
from NetConf import create_model \
from trainingPipeline import full_train \ 
from trainingPipeline import train,test,plot_signal \
import pandas as pd \
from torch.utils.tensorboard import SummaryWriter \
import torch \
import os \
import numpy as np \
import torchaudio \
import torch \
import torch.nn as nn \
import torch.nn.functional as F \
from torchvision import models \

#TRAINING FUNCTION 
#1)Требования к файлам
#СSV файл должен выглядеть также, как выглядит TDF.сsv. Только колонки должны называться path и classID 
#Аудио файлы должны быть формата .wav, дополнительных преобразований не требуется 

csv_path='TDF.csv' #путь к csv \
audio_path='wavs' #путь к аудио \
csv_path_t=csv_path #путь к тестовому csv \
audio_path_t=audio_path #путь к тестовому аудио \
save_pth='test_save_pth' #куда сохранить параметры \
deviceU=True #использовать ли GPU \
testT=True #Проводить ли тестирование \ 
save=True #сохранять ли веса \
#2)То, что влияет непосредственно на сетку \
batch_size=8 #Не думаю , что вам нужно это менять. Пока количество ваших данных очень мало - оставьте таким, когда размер датасета перевалит за 8к (Если перевалит) , то можно попробовать увеличить до 16 \
n_classes=3 #количество классов. Если в датасете классы - [0,1,2],то n должно быть равно 3 и тп. Надеюсь логика понятна \
n_epochs=10 #количество тренировочных итераций. Cитуация та же, пока данных мало - не трогать, как станет больше - можно немного увеличить (15\12) \
#3)Сама функция \
#full_train(csv_path,audio_path,8,3,10,save_pth,csv_path_t,audio_path_t,deviceU,testT,save) --> Находится в  trainingPipeline.py \

#VOICE_TO_TEXT FUNCTION

#1)Отправлять путь к .wav файлу \
#2)сама функция tts(path) --> находится в speech-to-text.py \

#ФУНКЦИЯ ДЛЯ ОПРЕДЕЛЕНИЯ ГОВОРЯЩЕГО

#1)аудио файлы должны быть .wav \
#2)cама функция prediction(param_pth,audio_pth,n_classes,n_epochs,useD=True) --> находится в GetResult.py \
#param_pth - путь к параметрам, n_classes и n_epochs - должны использоваться такие же, как и при обучении,
useD - использовать GPU или нет
#возвращает ID класса \
