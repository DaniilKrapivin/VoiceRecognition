from CreateDS import SoundDataset, create_train
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
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import PIL
import io

tensorboard_writer = SummaryWriter('./tensorboard_logs')
def plot_signal(signal, title, cmap=None):
    fig = plt.figure()
    if signal.ndim == 1:
        plt.plot(signal)
    else:
        plt.imshow(signal, cmap=cmap)
    plt.title(title)

    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='jpeg')
    plot_buf.seek(0)
    plt.close(fig)
    return ToTensor()(PIL.Image.open(plot_buf))


def train(model,optimizer,criterion,scheduler,epoch,loader,device,log_interval=10,debug_interval=25):
    model.train()
    for batch_idx, (sounds, sample_rate, inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iteration = epoch * len(loader) + batch_idx
        if batch_idx % log_interval == 0:  # print training stats
            print('Эпоха: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(loader.dataset),
                          100. * batch_idx / len(loader), loss))
            tensorboard_writer.add_scalar('Loss на обучении', loss, iteration)
            tensorboard_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], iteration)

        if batch_idx % debug_interval == 0: 
            for n, (inp, pred, label) in enumerate(zip(inputs, predicted, labels)):
                series = 'label_{}_pred_{}'.format(label.cpu(), pred.cpu())
                tensorboard_writer.add_image('Тренировочные MelSpectrogram сэмплы/{}_{}_{}'.format(batch_idx, n, series),
                                             plot_signal(inp.cpu().numpy().squeeze(), series, 'hot'), iteration)


def test(model,epoch,t_loader,n_classes,device,log_interval=10,debug_interval=25):
    model.eval()
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    with torch.no_grad():
        for idx, (sounds, sample_rate, inputs, labels) in enumerate(t_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(len(inputs)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

            iteration = (epoch + 1) * len(t_loader)
            if idx % debug_interval == 0: 
                for n, (sound, inp, pred, label) in enumerate(zip(sounds, inputs, predicted, labels)):
                    series = 'label_{}_pred_{}'.format(label.cpu(), pred.cpu())
                    tensorboard_writer.add_audio('Тестовые аудио сэмплы/{}_{}_{}'.format(idx, n, series),
                                                 sound, iteration, int(sample_rate[n]))
                    tensorboard_writer.add_image('Тестовые MelSpectrogram сэмплы/{}_{}_{}'.format(idx, n, series),
                                                 plot_signal(inp.cpu().numpy().squeeze(), series, 'hot'), iteration)

    total_accuracy = 100 * sum(class_correct) / sum(class_total)
    print('[Итерация {}] Точность на {} тестовых сэмплов: {}%\n'.format(epoch, sum(class_total), total_accuracy))
    tensorboard_writer.add_scalar('конечная точность', total_accuracy, iteration)



def full_train(csv_path,audio_path,batch_size,n_classes,n_epochs,
               save_pth,csv_path_t,audio_path_t,deviceC=True,testT=True,save=True):
    if deviceC and torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
        
    if testT:
        t_loader=create_train(csv_path_t, audio_path_t, batch_size=batch_size, return_a=True)
    loader = create_train(csv_path, audio_path, batch_size=batch_size, return_a=False)
    model, optimizer, scheduler, criterion = create_model(n_classes, n_epochs, device=True)
    model.to(device)
    if testT:
        for epoch in range(n_epochs):
            train(model, optimizer, criterion, scheduler, epoch, loader,device)
            test(model, epoch, t_loader, n_classes,device)
            scheduler.step()
    else:
        for epoch in range(n_epochs):
            train(model, optimizer, criterion, scheduler, epoch, loader,device)
            scheduler.step()
    if save:
        torch.save(model.state_dict(), save_pth)
