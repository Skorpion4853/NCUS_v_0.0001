import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

from dataset import EmotionDataset
from NN_class import EmotionClassifier

# Загружаем датасет
train_df = pd.read_csv('train_splitted.csv')
val_df = pd.read_csv('val_splitted.csv')

# "DATA PREPROCESSING" - после этих слов аниме продлилось на ещё 1000 серий
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
max_len = 256

### Обучаем + сохраняем результат (модель + веса)
d_train = EmotionDataset()
train_dataloader = data.DataLoader(d_train, batch_size=4, shuffle=True, num_workers=4)

val_data = EmotionDataset()
val_dataloader = data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

# Выбираем устройство
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Создаем объект модели, оптимизатора и тд.
epochs = 100

model = EmotionClassifier(n_classes=6).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)
loss_func = nn.CrossEntropyLoss().to(device)

def train_epoch(model, optimizer, scheduler, x, y, lm_count, loss_mean):
    model = model.train()
    pred = torch.argmax(model(x).to(device))
    loss = loss_func(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    lm_count += 1
    loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean

    return loss_mean

def valid_model(model, dataloader, loss_func):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for x, y in dataloader:

            pred = model(x).to(device)
            pred_item = torch.argmax(pred).item()
            label = torch.argmax(y).item()
            loss = loss_func(pred, y)

            correct_predictions += torch.sum(pred_item == label)
            losses.append(loss.item())

    val_acc = correct_predictions.double() / len(dataloader)
    val_loss = np.mean(losses)

    return val_acc, val_loss

#Запуск процесса обучения
for epoch in range(epochs):
    train_tqdm = tqdm(train_dataloader, leave=True)
    lm_count = 0
    loss_mean = 0

    for x, y in train_tqdm:
        loss_mean = train_epoch(model, optimizer, scheduler, x, y, lm_count, loss_mean)
        train_tqdm.set_description(f"Epoch [{epoch + 1}/{epochs}], loss_mean={loss_mean:.3f}")

    # Валидация + сохранение норм моделей
    best_acc = 0
    val_acc, val_loss = valid_model(model, val_dataloader, loss_func)

    if val_acc>best_acc:
        save_data = [model, model.state_dict()]
        torch.save(save_data, f=f'models/emotion_classifier_epoch{epoch+1}')


