import pandas as pd
from collections import Counter
import numpy as np
import torch

def get_targets(df):
    tensor = torch.tensor(df.to_numpy())
    targets_df = pd.DataFrame()
    for i in range(len(tensor)):
        targets_df = pd.concat([pd.DataFrame([str(torch.argmax(tensor[i]).item())], columns=['a']), targets_df], ignore_index=True)

    return targets_df

'''
df = get_targets(pd.read_csv('train_splitted.csv').iloc[:, 2:8])
df1 = get_targets(pd.read_csv('val_splitted.csv').iloc[:, 2:8])

print (df.groupby('a').size())
print (df1.groupby('a').size())
'''

df = pd.read_csv('val_splitted.csv')


while df.loc[:,'neutral'].sum() > 1000:
        df = df.drop(df[df['neutral'] == 1].index[0],axis=0)

while df.loc[:,'anger'].sum() > 1000:
        df = df.drop(df[df['anger'] == 1].index[0], axis=0)


df1 = get_targets(df.iloc[:, 2:8])
print (df1.groupby('a').size())

print(f'{df}')

df.to_csv('val111.csv', index=False)




df = pd.read_csv('train_splitted.csv')


while df.loc[:,'neutral'].sum() > 3001:
        df = df.drop(df[df['neutral'] == 1].index[0],axis=0)

while df.loc[:,'anger'].sum() > 3001:
        df = df.drop(df[df['anger'] == 1].index[0], axis=0)


df1 = get_targets(df.iloc[:, 2:8])
print (df1.groupby('a').size())

print(f'{df}')

df.to_csv('train111.csv', index=False)