import os

import torch
from transformers import BertTokenizer
from NN_class import EmotionClassifier

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
os.chdir('/Users/atlas/Downloads/NCUS_v_0.0001-Chaihana-branch/emotion_classifier_train')

torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())
# Загружаем данные модели (0 - архитектура (класс), 1 - веса модели, 2 - dict c показателями модели)
model_params = torch.load('models/emotion_classifier_epoch[2].tar')
print('model file loaded')

# Создаем объект модели
model = model_params[0]
model.load_state_dict(model_params[1])
print('model object created')

# Тестируем модель на рандомном наборе текста введенном вручную
text = ('УРА я наконец-то дошел до этого этапа реализации проекта, я так счастлив! '
        'это же просто прекрасно! скоро я завершу работу и у меня будет столько свободного времени, '
        'ведь у меня же нет ещё 4 проектов висящих на мне!')
# anger - 0, excitement - 1,fear - 2,optimism - 3,sadness - 4,neutral - 5
emotions = {0:'anger', 1:'excitement', 2:'fear', 3:'optimism', 4:'sadness', 5:'neutral'}

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
max_len = 256
print('text, tokenizer done')

encoding = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_len,
                                 return_token_type_ids=False, pad_to_max_length=True,
                                 return_attention_mask=True, return_tensors='pt')
print(f'text tokenized')

model.eval()

with torch.no_grad():
    outputs = model(input_ids=encoding['input_ids'].flatten(), attention_mask=encoding['attention_mask'].flatten())
    print('got model outputs')
    pred = torch.argmax(outputs).item()
    print('prepared prediction')
    print(f'out tensor:\n{outputs}\n\nmodel prediction: {pred} | {emotions[pred]}')