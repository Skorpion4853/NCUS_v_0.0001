from pymystem3 import Mystem
import torch
import transformers
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
    hotkey_list = ['admiration','amusement','anger',
                   'annoyance','approval','caring',
                   'confusion','curiosity','desire',
                   'disappointment','disapproval','disgust',
                   'embarrassment','excitement','fear',
                   'gratitude','grief','joy',
                   'love','nervousness','optimism',
                   'pride','realization','relief',
                   'remorse','sadness','surprise',
                   'neutral']
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

    emot_model_checkpoint = torch.load('models/sentiment_classifier_GRU_epoch[31].tar',weights_only=False)
    emot_model = emot_model_checkpoint['model']
    emot_model.load_state_dict(emot_model_checkpoint['weights'])
    navec_emb = emot_model_checkpoint['navec']

    return device, t5_tokenizer, t5_model, emot_model, navec_emb

def load_voice_engine():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 200)
    engine.setProperty('voice', voices[0].id)

    return engine