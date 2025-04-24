import torch
import re
from RNNclass import SentimentClassifier

loaded = torch.load(f='models/sentiment_classifier_epoch[10].tar', weights_only=False)
navec_emb = loaded['navec']

model = SentimentClassifier(in_features=300, out_features=28)
model.load_state_dict(loaded['weights'])
model.eval()



print('Model loaded')

texts_val = ['глупо то что делает глупость - мама гампа',
         'это жестоко говорить человеку страдающему депрессией',
         'я конечно нет',
         'лмао почему каждый фанат sw должен быть таким мелодраматичным',
         'я чувствую что [имя] сегодня вечером проведет отличный ответный матч']

texts = ['я арбуз',
         'ненавижу тебя убирайся из этого дома',
         'вы только посмотрите какое уродливое личико',
         'обожаю этот фильм',
         'наконец-то мы встретились как же я рад этому нет ничего лучше чем это']

# re.sub(r'[^А-яA-z- ]', '', str(self.data.iloc[item, 0]).replace('\ufeff', '').replace('\n', ' ')).lower()

for i in range(len(texts)):
    texts[i] = re.sub(r'[^А-яA-z- ]', '', str(texts[i]).replace('\ufeff', '').replace('\n', ' ')).lower()

for i in range(len(texts_val)):
    texts_val[i] = re.sub(r'[^А-яA-z- ]', '', str(texts_val[i]).replace('\ufeff', '').replace('\n', ' ')).lower()

# words = [word for word in str(self.data.iloc[item, 0:1]).split(' ') if word in self.navec_emb]
# text = torch.vstack([torch.tensor(self.navec_emb[word]) for word in words])

for text in texts_val:
    words = [word for word in str(text).split(' ') if word in navec_emb]
    in_text = torch.vstack([torch.tensor(navec_emb[word]) for word in words])

    with torch.no_grad():
        out = model(in_text)

    pred = torch.argmax(torch.softmax(out, 0))

    print(f'text >> {text}\nout tensor >> {out}\nprediction >> {pred}')
