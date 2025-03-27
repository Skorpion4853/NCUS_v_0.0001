from pymystem3 import Mystem
import torch
import transformers
import tensorflow as tf
import pickle
from nltk.corpus import stopwords
import pyttsx3

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

def lemmatiz(text, lst):
    m = Mystem()
    lemmas = m.lemmatize(text)
    lst.append("".join(lemmas))

def load_models():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    t5_tokenizer = transformers.GPT2Tokenizer.from_pretrained("SiberiaSoft/SiberianPersonaFred-2")
    t5_model = transformers.T5ForConditionalGeneration.from_pretrained("SiberiaSoft/SiberianPersonaFred-2")
    emot_model = tf.keras.models.load_model('model_classifire.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        token = pickle.load(handle)
    stop_words = set(stopwords.words('russian'))

    return device, t5_tokenizer, t5_model, emot_model, token, stop_words

def load_voice_engine():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 200)
    engine.setProperty('voice', voices[0].id)

    return engine