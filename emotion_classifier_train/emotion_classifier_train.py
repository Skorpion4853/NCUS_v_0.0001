import pandas as pd
import numpy as np
import torch.nn as nn
import torch.utils.data as data

# Загружаем датасет
train_df = pd.read_csv('train.csv')

# Создаем класс датасета
class EmotionDataset(data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, item):
        pass

# Создаем модель
class Emotion_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding()
        self.rls = nn.Sequential(nn.RNN(),
                                nn.Tanh())
        self.dls = nn.Sequential(nn.Linear(in_features=32, out_features=6),
                                 nn.ReLU())
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embed(x)
        x = self.rls(x)
        x = self.dls(x)
        x = self.softmax(x)

        return x

# Обучаем + сохраняем результат (модель + веса)
d_train = EmotionDataset()
train_data = data.DataLoader(d_train, batch_size=4, shuffle=True)


model = Emotion_Classifier()
epochs = 100
model.train()

for epoch in range(epochs):
    pass