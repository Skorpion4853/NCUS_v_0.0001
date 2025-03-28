import pandas as pd
import numpy as np
#import torch.nn as nn



# Загружаем датасет
train_df = pd.read_csv('dirty_train.csv')

print(train_df)

# Обрабатываем датасет
def clear_dataset(data):
    ids = []
    incorrect_columns = ['author', 'subreddit', 'link_id',
                         'id', 'parent_id', 'created_utc',
                         'rater_id']
    correct_columns = ['ru_text', 'text', 'anger', 'fear', 'excitement', 'sadness', 'optimism', 'neutral']
    data.drop(incorrect_columns, axis=1, inplace=True)
    columns = data.columns.tolist()

    for column in columns:
        if column not in correct_columns:
            indexes = data[ data[column] == 1].index
            data.drop(indexes, inplace=True)

    for column in columns:
        if column not in correct_columns:
            data.drop(column, axis=1, inplace=True)


    return data

clean_dataset = clear_dataset(train_df)
print(f'Columns:\n{clean_dataset.columns}\nlen: {len(clean_dataset)}')
clean_dataset.to_csv('train.csv', index=False)

'''

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

'''