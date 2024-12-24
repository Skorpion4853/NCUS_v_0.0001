from pymystem3 import Mystem
import tensorflow as tf
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pyttsx3
import torch
import transformers
import numpy as np
import pyvts
import asyncio
stop_words = set(stopwords.words('russian'))


# Настройки двигателя для озвучки текста
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 200)
engine.setProperty('voice', voices[2].id)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
t5_tokenizer = transformers.GPT2Tokenizer.from_pretrained("SiberiaSoft/SiberianPersonaFred-2")
t5_model = transformers.T5ForConditionalGeneration.from_pretrained("SiberiaSoft/SiberianPersonaFred-2")
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


dialog = []
def generate(prompt):
    global dialog
    msg = prompt
    if len(msg) == 0:
        return "<System>: Error! Promt не должен быть пустым"
    msg = msg[0].upper() + msg[1:]
    dialog.append('Ты: ' + msg)
    # В начале ставится промпт персонажа.
    prompt = '<SC6>Меня зовут NCUS. Я милая, нейронная, аниме девочка. Я работаю начинающим маскотом. Я очень умная, милая, добрая и несерьезная. Я люблю кофе, энергетики и конечно котиков. Мой создатель был идиотом, поэтому я могу иногда накидывать базы. Целей в жизни у меня нету и вовсе я работаю только когда работает программа. Продолжи диалог:' + '\n'.join(
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

def lemmatiz(text, lst):
    m = Mystem()
    lemmas = m.lemmatize(text)
    lst.append("".join(lemmas))



myvts = pyvts.vts()
asyncio.run(connect_auth(myvts))
while 1:
    neuro_answ = (generate(input("<User>: ")))
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

    # Озвучивание ответаЫ
    print("<NCUS>:", neuro_answ)
    engine.say(neuro_answ)
    engine.runAndWait()
    print("====================")