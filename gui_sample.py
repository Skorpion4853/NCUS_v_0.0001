# -*- coding: utf-8 -*-
from funcs import *
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pyvts
import asyncio
import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from test_ui import Ui_MainWindow

# Настройки двигателя для озвучки текста
engine = load_voice_engine()

# Загружаем модель
device, t5_tokenizer, t5_model, emot_model, token, stop_words = load_models()

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
        lemmatiz(neuro_answ, user_list)

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
        self.ui.chat.setPlainText('nueroSama -> ' + neuro_answ + '\n' + self.ui.chat.toPlainText())
        self.ui.chat.setPlainText('user -> ' + self.ui.msg.toPlainText() + '\n' + self.ui.chat.toPlainText()+ '\n')
        print(neuro_answ)
        engine.say(neuro_answ)
        engine.runAndWait()
        print("====================")

if __name__== "__main__":
    app = QApplication(sys.argv)
    myapp = Test()
    myapp.show()
    app.exec()
