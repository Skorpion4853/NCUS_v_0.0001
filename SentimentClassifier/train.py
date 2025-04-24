from Dataset import SentimentDataset
from RNNclass import SentimentClassifier, SentimentClassifierGRU
from navec import Navec
import torch.utils.data as data
import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

mac_path = '/Users/atlas/Downloads/NCUS_v_0.0001-Chaihana-branch/SentimentClassifier'
pc_path = 'D:/NCUS_v_0.0001-Chaihana-branch/SentimentClassifier'
os.chdir(pc_path)

print('Changed dir')

path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

train_data = SentimentDataset(path='datasets/train.csv',navec_emb=navec)
train_dataloader = data.DataLoader(train_data, batch_size=1, shuffle=True)

val_data = SentimentDataset(path='datasets/val.csv',navec_emb=navec)
val_dataloader = data.DataLoader(val_data, batch_size=1, shuffle=False)

print('CSV loaded')
print('Dataloaders done')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = SentimentClassifierGRU(300, 28).to(device)
print('Model object created')

epochs = 100  # как-то раз и невзначай сунул 100 эпох по 40 минут каждая на обучение
best_acc = 100000

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

loss_func = nn.CrossEntropyLoss()

print('optim, scheduler, loss_func done')
print('\n-----------------------\nStarting training')

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}\n------------')

    train_tqdm = tqdm(train_dataloader, leave=True)
    loss_mean = 0
    lm_count = 0

    model.train()
    for x,y in train_tqdm:
        out = model(x.to(device))
        loss = loss_func(out, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"loss_mean={loss_mean:.3f}")

    print(f'Epoch[{epoch + 1}/{epochs}] loss_mean={loss_mean}')

    # Валидация + сохранение норм моделей

    model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for x, y in val_dataloader:
            outputs = model(x.to(device))

            loss = loss_func(outputs, y.to(device))
            losses.append(loss.item())

    val_loss = np.mean(losses)

    print(f'Epoch[{epoch + 1}/{epochs}] val_loss={val_loss}')

    if val_loss < best_acc:
        save_data = {'weights':model.to('cpu').state_dict(),
                     'model':model.to('cpu'),
                     'navec':navec,
                     'stats':{'train_loss_mean':loss_mean,
                              'val_loss_mean':val_loss}
                     }
        torch.save(save_data, f=f'models/sentiment_classifier_GRU_epoch[{epoch + 1}].tar')
        best_acc = val_loss
        print('Model saved!')
    model.to(device)