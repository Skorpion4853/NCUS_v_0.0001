import pandas as pd
import numpy as np
import torch.nn as nn



# Загружаем датасет
# Я 1.5 ПАРЫ УДАЛЯЛ НЕНУЖНЫЕ СТОЛБЦЫ ИЗ ДАТАСЕТА Я ТОГО ВСË
train_df = pd.read_csv('train.csv')

# Создаем модель
class Emotion_Classifier(nn.Module):
    def __init__(self):
        self.LSTM = nn.LSTM(input_size=1000, hidden_size=32)
        self.layer1 = nn.Linear(in_features=32, out_features=6)

    def forward(self, x):
        x = self.LSTM(x)
        x = self.layer1(x)

        return x

# Обучаем + сохраняем результат (модель + веса)
model = Emotion_Classifier()
epochs = 100
model.train()

for epoch in range(epochs):
    pass
