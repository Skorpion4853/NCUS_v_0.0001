import torch
import pandas as pd
from torchtext.data.utils import get_tokenizer
from navec import Navec
from slovnet.model.emb import NavecEmbedding

'''
tokenizer = get_tokenizer(language='ru', tokenizer=None)

text = 'Я арбуз 322, гойда,   дота!!!!????'

tokens = tokenizer(text)

path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

a = navec['арбуз']

ids = [navec[word] for word in tokens if word in navec]



print(f'{tokens}\n')

df = pd.read_parquet("hf://datasets/seara/ru_go_emotions/raw/train-00000-of-00001-86de8ef1d0ae28df.parquet")

df.to_csv('train.csv', index=False)
'''

'''
#ru_text,text,id,,admiration,amusement,anger,annoyance,approval,caring,confusion,curiosity,desire,disappointment,disapproval,disgust,embarrassment,excitement,fear,gratitude,grief,joy,love,nervousness,optimism,pride,realization,relief,remorse,sadness,surprise,neutral

df = pd.read_csv('train.csv')

df.drop(columns=['text','id','author','subreddit','link_id','parent_id','created_utc','rater_id','example_very_unclear'], inplace=True)

df.to_csv('train_.csv', index = False)

#Я ЗАБИЛ ОГРОМНЫЙ на уменьшение labels, крч, будет 28 эмоций, мне поххххххх
'''

