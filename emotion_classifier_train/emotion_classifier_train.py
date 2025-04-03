import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import EmotionDataset
from NN_class import EmotionClassifier
from train_funcs import train_epoch, valid_model

# Загружаем датасет - как же я его манал
train_df = pd.read_csv('train_splitted.csv')
val_df = pd.read_csv('val_splitted.csv')

### "DATA PREPROCESSING" - после этих слов аниме продлилось на ещё 1000 серий
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
max_len = 256

### Обучаем + сохраняем результат (модель + веса) - нихрена он не обучился
train_data = EmotionDataset(train_df.loc[:, 'ru_text'],train_df.iloc[:, 2:8],tokenizer, max_len)    #анекдот дня: заходит как-то iloc и loc в бар
train_dataloader = data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)           # iloc говорит loc'у: "биба я название столбца не чувствую"
                                                                                                    # а loc ему в ответ "боба у тебя его нет"
val_data = EmotionDataset(train_df.loc[:, 'ru_text'],train_df.iloc[:, 2:8],tokenizer, max_len)
val_dataloader = data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)


# Выбираем устройство
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     # появился как-то в зоне tensorflow клятый
                                                                                        # подходит к палатке и говорит "дооооступ... к... GPU..."
                                                                                        # и кто доступ к GPU не давал, тому он ночью git сносил
# Создаем объект модели, оптимизатора и тд.
epochs = 100                                                                            # как-то раз и невзначай сунул 100 эпох по 40 минут каждая на обучение

model = EmotionClassifier(n_classes=6).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

loss_func = nn.CrossEntropyLoss().to(device)


#Запуск процесса обучения
for epoch in range(epochs):
    train_tqdm = tqdm(train_dataloader, leave=True)
    lm_count = 0
    loss_mean = 0

    for x, y in train_tqdm:
        loss_mean = train_epoch(model, optimizer, scheduler, loss_func, device, x, y, lm_count, loss_mean)
        train_tqdm.set_description(f"Epoch [{epoch + 1}/{epochs}], loss_mean={loss_mean:.3f}")

    # Валидация + сохранение норм моделей
    best_acc = 0
    val_acc, val_loss = valid_model(model, val_dataloader, loss_func, device)

    if val_acc>best_acc:
        save_data = [model, model.state_dict()]
        torch.save(save_data, f=f'models/emotion_classifier_epoch{epoch+1}')


