import os
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

if __name__ == '__main__':
    os.chdir('D:/NCUS_v_0.0001-Chaihana-branch/emotion_classifier_train')

    print('Changed dir')

    # Загружаем датасет - как же я его манал
    train_df = pd.read_csv('train_splitted.csv')
    val_df = pd.read_csv('val_splitted.csv')

    print('CSV loaded')

    ### "DATA PREPROCESSING" - после этих слов аниме продлилось на ещё 1000 серий
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    max_len = 256

    # МОЗГОВ У МЕНЯ НЕТ, ЗАТО ЕСТЬ ИДЕЯ!!!
    # ALERT!!!!!! КОСТЫЛЬНО-ОРИЕНТИРОВАННОЕ ПРОГРАММИРОВАНИЕ!!!!!!!!!!!!!!
    # anger - 0, excitement - 1,fear - 2,optimism - 3,sadness - 4,neutral - 5
    '''
    def get_targets(df):
        tensor = torch.tensor(df.to_numpy())
        targets_df = pd.DataFrame()
        for i in range(len(tensor)):
            targets_df = pd.concat([pd.DataFrame([torch.argmax(tensor[i]).item()]), targets_df], ignore_index=True)

        return targets_df
    '''
    # Я искренне извинясь перед всеми богами мира за говнокодерство свыше, это единичное недоразумение вызванное случайным стечением обстоятельств и не было написанно мной

    ### Обучаем + сохраняем результат (модель + веса) - нихрена он не обучился
    train_data = EmotionDataset(train_df.loc[:, 'ru_text'].to_numpy(),train_df.iloc[:, 2:8].to_numpy(),tokenizer, max_len)    #анекдот дня: заходит как-то iloc и loc в бар
    train_dataloader = data.DataLoader(train_data, batch_size=64)           # iloc говорит loc'у: "биба я название столбца не чувствую"
                                                                                                        # а loc ему в ответ "боба у тебя его нет"
    val_data = EmotionDataset(val_df.loc[:, 'ru_text'].to_numpy(),val_df.iloc[:, 2:8].to_numpy(),tokenizer, max_len)
    val_dataloader = data.DataLoader(val_data, batch_size=16)

    print('Dataloaders done')

    # Выбираем устройство
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     # появился как-то в зоне tensorflow клятый
                                                                                            # подходит к палатке и говорит "дооооступ... к... GPU..."
                                                                                            # и кто доступ к GPU не давал, тому он ночью git сносил
    # Создаем объект модели, оптимизатора и тд.
    epochs = 500                                                                            # как-то раз и невзначай сунул 100 эпох по 40 минут каждая на обучение
    best_acc = 100000

    model = EmotionClassifier(n_classes=6).to(device)

    print('Model object created')

    optimizer = AdamW(model.parameters(), lr=2e-5)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_func = nn.CrossEntropyLoss().to(device)

    print('optim, scheduler, loss_func done')
    print('\n-----------------------\nStarting training')

    #Запуск процесса обучения
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}\n------------')
        train_acc, loss_mean = train_epoch(model, optimizer, scheduler,
                                loss_func, train_dataloader, device)
        print(f'Epoch[{epoch+1}/{epochs}] train_acc={train_acc} loss_mean={loss_mean}')

        # Валидация + сохранение норм моделей

        val_acc, val_loss = valid_model(model, val_dataloader, loss_func, device)

        print(f'Epoch[{epoch + 1}/{epochs}] val_acc={val_acc} val_loss={val_loss}')

        if val_loss<best_acc:
            save_data = [model, model.state_dict(),
                         {'model_train_acc':train_acc, 'model_train_loss_mean':loss_mean,
                          'model_valid_acc':val_acc, 'model_valid_loss':val_loss}]
            torch.save(save_data, f=f'models/emotion_classifier_epoch[{epoch+1}].tar')
            best_acc = val_loss
            print('Model saved!')

# ЭТОТ КРЕТИН НАЧАЛ ОБУЧАТЬСЯ, 2:28 НОЧИ Я ТОГО ВСЕ
# ЕСЛИ У МЕНЯ ЗАБЬЕТСЯ ПАМЯТЬ НА КОМПЕ ОТ СОХРАНЕННЫХ МОДЕЛЕЙ Я ВЗОРВУСЬ НАХУЙ
# Я МАНАЛ ФИКСИТЬ БАГИ, ТАК ЕЩЁ И ГАЙД БЫЛ С ЛЕГАСИ КОДОМ, КОТОРЫЙ ПЕРЕПИСЫВАТЬ ПРИШЛОСЬ
# А БЕЗ ВАРНИНГОВ ОНО ЕЩЁ И НЕ ЗАПУСТИЛОСЬ А ВРЕМЕНИ НА ФИКС НЕТ
# так ещё и оказалось, я не могу быстро проверить обучается он или нет, тк у меня на видюхе на большой batch_size памяти не хватает
# и судя по всему на эпоху будет уходить охренеть сколько времени (tqdm не получилось успеть подкрутить к легаси коду из чертового гайда)

# Выводы: индусы - нехорошие, я - говнокодер, моя видюха - RIP 04.04.2025 2:40, а обучаться я его оставляю на ночь даже не зная в порядке ли все