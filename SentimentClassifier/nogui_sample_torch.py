# -*- coding: utf-8 -*-
from click import prompt

from funcs import *
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pyvts
import asyncio
import re
import sys

# Настройки двигателя для озвучки текста
engine = load_voice_engine()

# Загружаем модель
device, t5_tokenizer, t5_model, emot_model, navec_emb = load_models()

dialog = []
def generate(prompt):
    global dialog
    msg = prompt
    if len(msg) == 0:
        return "<System>: Error! Promt не должен быть пустым"
    msg = msg[0].upper() + msg[1:]
    dialog.append('Ты: ' + msg)
    # В начале ставится промпт персонажа.
    prompt = "<SC6>Меня зовут NCUS. Я милая, нейронная, аниме девочка. Я работаю начинающим маскотом. Я очень умная, милая, добрая и несерьезная. Я люблю кофе, энергетики и конечно котиков. Мой создатель был идиотом, поэтому я могу иногда накидывать базы. Целей в жизни у меня нету и вовсе я работаю только когда работает программа. Продолжи диалог:" + '\n'.join(
        dialog) + '\nЯ: <extra_id_0>'
    input_ids = t5_tokenizer(prompt, return_tensors='pt').input_ids
    out_ids = t5_model.generate(input_ids=input_ids.to(device), do_sample=True, temperature=0.9, max_new_tokens=512,
                                top_p=0.85,
                                top_k=2, repetition_penalty=1.2)
    t5_output = t5_tokenizer.decode(out_ids[0][1:])
    if '</s>' in t5_output:
        t5_output = t5_output[:t5_output.find('</s>')].strip()
    t5_output = t5_output.replace('<extra_id_0>', '').strip()
    t5_output = t5_output.split('Собеседник')[0].strip()
    dialog.append('Я: ' + t5_output)
    return t5_output

myvts = pyvts.vts()
asyncio.run(connect_auth(myvts))

while True:
    prompt = input('Введите промпт: ')
    print(f'Промпт: {prompt}')

    neuro_answ = (generate(prompt))

    # пред-обработка данных для прогнозирования эмоции
    emot_text = re.sub(r'[^А-яA-z- ]', '', str(neuro_answ).replace('\ufeff', '').replace('\n', ' ')).lower()
    words = [word for word in str(emot_text).split(' ') if word in navec_emb]
    emot_input = torch.vstack([torch.tensor(navec_emb[word]) for word in words])

    # Прогназирование эмоции
    with torch.no_grad():
        out = emot_model(emot_input)
        pred = torch.argmax(torch.softmax(out, 0)).item()
    asyncio.run(trigger(myvts, pred))

    # Озвучивание ответа
    print(neuro_answ)
    engine.say(neuro_answ)
    engine.runAndWait()
    print(f'emotion >> {pred}')
    print("====================")

