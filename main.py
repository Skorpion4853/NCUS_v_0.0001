from pymystem3 import Mystem
import tensorflow as tf
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pyttsx3
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, pipeline
import numpy as np
import pyvts
import asyncio
import sys

import keyboard
from PyQt5.uic.Compiler.qtproxies import QtWidgets
from click import prompt
from pynput.keyboard import Key, Listener

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QFontDatabase
from PyQt5 import QtCore, QtGui, QtWidgets
from test_ui import Ui_MainWindow

stop_words = set(stopwords.words('russian'))

pipe = pipeline("text2text-generation", model="SiberiaSoft/SiberianFredT5-instructor")

# Настройки двигателя для озвучки текста
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 200)
engine.setProperty('voice', voices[2].id)

tokenizer = AutoTokenizer.from_pretrained("SiberiaSoft/SiberianFredT5-instructor")
model = AutoModelForSeq2SeqLM.from_pretrained("SiberiaSoft/SiberianFredT5-instructor")
model.eval()
emot_model = tf.keras.models.load_model('model_classifire.keras')
with open('tokenizer.pickle', 'rb') as handle:
    token = pickle.load(handle)


async def connect_auth(myvts):
    ''' functions to get authenticated '''
    await myvts.connect()
    await myvts.request_authenticate_token()
    await myvts.request_authenticate()
    await myvts.close()

async def trigger(myvts, emotion):
    ''' function to trigger hotkey '''
    await myvts.connect()
    await myvts.request_authenticate()
    hotkey_list = ['Without_Emotion', 'Happy', 'Sad', 'surprise', 'fear', 'Angry']
    send_hotkey_request = myvts.vts_request.requestTriggerHotKey(hotkey_list[emotion])
    await myvts.request(send_hotkey_request)  # send request to play 'My Animation 1'
    await myvts.close()

def generate(prompt):
    data = tokenizer('<SC6>' + prompt + '\nОтвет: <extra_id_0>', return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data, do_sample=True, temperature=0.2, max_new_tokens=512, top_p=0.95, top_k=5, repetition_penalty=1.03,
        no_repeat_ngram_size=2
    )[0]
    out = tokenizer.decode(output_ids.tolist())
    out = out.replace("<s>", "").replace("</s>", "")
    return out


def lemmatiz(text, lst):
    m = Mystem()
    lemmas = m.lemmatize(text)
    lst.append("".join(lemmas))



myvts = pyvts.vts()
asyncio.run(connect_auth(myvts))

class Test(QMainWindow):
    def __init__(self):
        super(Test, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.btn_snd.clicked.connect(lambda: self.snd())

    def snd(self):
        print('Промпт: '+self.ui.msg.toPlainText())
        neuro_answ = (generate(self.ui.msg.toPlainText()))
        user_list = []

        # пред-обработка данных для прогнозирования эмоции
        lemmatiz(neuro_answ[17:], user_list)

        stop_words = set(stopwords.words('russian'))
        tokens = word_tokenize(user_list[0].replace("'", "").replace("\n", ""), 'russian')
        clear_user_input = []
        for word in tokens:
            if word not in stop_words:
                clear_user_input.append(word)
        user_x = token.texts_to_sequences(clear_user_input)
        user_x = pad_sequences(user_x, maxlen=15690, padding='post', truncating='post')
        user_x = token.texts_to_matrix([clear_user_input])

        # Прогназирование эмоции
        y_pred = emot_model.predict(user_x)
        Emotion = np.where(y_pred[0] == y_pred.max())[0][0]
        asyncio.run(trigger(myvts, Emotion))

        # Озвучивание ответа
        self.ui.chat.setPlainText('nueroSama -> ' + neuro_answ[17:] + '\n' + self.ui.chat.toPlainText())
        self.ui.chat.setPlainText('user -> ' + self.ui.msg.toPlainText() + '\n' + self.ui.chat.toPlainText())
        print(neuro_answ[17:])
        engine.say(neuro_answ[17:])
        engine.runAndWait()
        print("====================")

if __name__== "__main__":
    app = QApplication(sys.argv)
    myapp = Test()
    myapp.show()
    app.exec()





