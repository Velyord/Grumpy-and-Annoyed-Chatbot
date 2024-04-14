print("\nААА!?")
print("Кой ми светна лампите!?")
print("Дай ми минутка да се съвзема...\n")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # премахване на излишни предупреждения от конзолата
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # премахване на излишни предупреждения от конзолата
import warnings
warnings.filterwarnings("ignore") # премахване на излишни предупреждения от конзолата

from keras.models import load_model
from keras.layers import Dense
from keras.models import Sequential
import pickle
import json
import numpy as np
import tensorflow as tf
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer() # за коренуване на думите

try:
    with open('intents.json', encoding='utf-8') as json_data: # utf-8 за ползване на кирилица
        intents = json.load(json_data) # зареждане и обработка на намеренията от файл intents.json
except FileNotFoundError:
    print("Грешка: файлът intents.json не е намерен. Моля, уверете се, че файлът съществува.")
    exit(1)

words = [] # отделни думи
tags = [] # тагове
wordToTag = [] # дума - таг
ignoreWords = ['?', '!', '.']

'''
Токенизирането включва разделяне на изреченията на отделни думи за по-нататъшен анализ, обработка и извличане на функции. 
Това позволява на компютрите да разбират и обработват човешкия език.
'''

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern) # разделяне на всяка дума в изречението
        words.extend(word)
        wordToTag.append((word, intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

'''
Разделянето на корени се използва за предварителна обработка на думите в изреченията.
Тази стъпка гарантира, че подобни думи се третират идентично по време на обучението, 
като по този начин се подобрява способността на модела да разбира и отговаря ефективно на потребителските заявки.
'''

words = [stemmer.stem(word.lower()) for word in words if word not in ignoreWords] # намиране на корена на думата, смаляване на главни букви и премахване на препинателни знаци
words = sorted(list(set(words))) # премахване на дубликати
tags = sorted(list(set(tags))) # премахване на дубликати

training = []
output = []
outputEmpty = [0] * len(tags)
maxBagLength = 0
trainingData = []
outputEmpty = [0] * len(tags) # лист от нули със същия размер като този на таговете 

'''
Представянето на чантата с думи се използва за преобразуване на изреченията в числови вектори за обучение на чатбота. 
Всяко изречение е представено като вектор, показващ наличието или отсъствието на думи от речника. 
Това представяне позволява на модела да научи асоциации между входните изречения и съответните им намерения или отговори.
'''

for doc in wordToTag:
    bag = [] 
    patternWords = doc[0]
    patternWords = [stemmer.stem(word.lower()) for word in patternWords] # намиране на корена на думата, смаляване на главни букви и премахване на препинателни знаци
    for word in words:
        bag.append(1) if word in patternWords else bag.append(0)        
    maxBagLength = max(maxBagLength, len(bag))
    outputRow = list(outputEmpty)
    outputRow[tags.index(doc[1])] = 1
    trainingData.append((bag, outputRow))

'''
Подготвяне на данните за обучение, като се преобразуват в масиви NumPy 
и се структурират във формат, подходящ за обучение на модела на невронната мрежа. 
Входните характеристики (trainX) и изходните етикети (trainY) се извличат от данните за обучение, 
Освен това графиката на TensorFlow се нулира, за да се осигури чист лист преди изграждането на модела на невронната мрежа.
'''

bags = [data[0] for data in trainingData]
outputRows = [data[1] for data in trainingData]
bagsNumpy = np.array(bags)
output_rows_np = np.array(outputRows)
training = np.column_stack((bagsNumpy, output_rows_np))
trainX = training[:, :-len(outputEmpty)]
trainY = training[:, -len(outputEmpty):]
tf.compat.v1.reset_default_graph()
print("Оф, още ли си тук? Момент...")

'''
Създава се проста невронна мрежа с предварителна връзка с помощта на „Keras Sequential“. 
Моделът се състои от входен слой с 8 неврона всеки и изходен слой със същия брой неврони като класовете в данните за обучение. 
След това моделът се обучава върху обучителния набор от данни за 666 епохи, 
като се използва оптимизаторът на Адам и функцията за категорична крос-ентропийна загуба.
'''

model = Sequential() # Sequential опростява процеса на изграждане на модели на невронни мрежи, като предоставя ясен и интуитивен интерфейс за добавяне на слоеве и дефиниране на потока от данни през мрежата.
model.add(Dense(8, input_dim=maxBagLength, activation='relu')) # ReLU е проста, но мощна функция за активиране, която помага на невронните мрежи да учат ефективно сложни модели.
model.add(Dense(8, activation='relu'))
model.add(Dense(len(trainY[0]), activation='softmax')) # softmax играе решаваща роля в преобразуването на необработените изходни резултати в значими вероятности, позволявайки на невронните мрежи да правят вероятностни прогнози в задачите за класификация.
model.compile(loss='categorical_crossentropy', # настройва се функцията за загуба, която да бъде оптимизирана по време на обучение, което позволява на невронната мрежа да се научи да прави точни прогнози в задачи за класификация на няколко класа.
              optimizer='adam', metrics=['accuracy']) # Adam е мощен алгоритъм за оптимизация, който съчетава адаптивни скорости на обучение, инерция и RMSProp за ефективно обучение на дълбоки невронни мрежи, което води до по-бързо сближаване и подобрена производителност при широк набор от задачи.

try: # Обучение
    model.fit(trainX, trainY, epochs=666, batch_size=8, verbose=0) # vebose=0 за премахване на излишна информация от конзолата
except Exception as e:
    print("Възникна грешка по време на обучението на модела: ", e)
    exit(1)

try: # Запазване на модела
    model.save('model.keras')
except Exception as e:
    print("Възникна грешка при запазването на модела: ", e)
    exit(1)

try: # Запазване на данните
    pickle.dump({'words': words, 'tags': tags, 'train_x': trainX,
                'train_y': trainY}, open("training_data", "wb"))
except Exception as e:
    print("Възникна грешка при запазване на данните: ", e)

try: # Зареждане на данните
    data = pickle.load(open("training_data", "rb"))
    words = data['words']
    tags = data['tags']
    trainX = data['train_x']
    trainY = data['train_y']
except Exception as e:
    print("Възникна грешка при зареждането на данните: ", e)
    exit(1)

try: # Зареждане на модела
    model = load_model('./model.keras')
except Exception as e:
    print("Възникна грешка при зареждането на модела: ", e)
    exit(1)

def clean_up_sentence(sentence):
    sentenceWords = nltk.word_tokenize(sentence) # Разделяне на изречението в отделни думи
    sentenceWords = [stemmer.stem(word.lower()) for word in sentenceWords] # Намиране на корена на думата
    return sentenceWords

ERROR_THRESHOLD = 0.25

def bow(sentence, words): # тази функция взема входно изречение, токенизира го и го почиства и след това го преобразува в двоично представяне на чанта от думи въз основа на присъствието или отсъствието на всяка дума в речника.
    sentenceWords = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentenceWords:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def classify(sentence):
    results = model.predict(np.array([bow(sentence, words)]))[0] # Прогноза на модела
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD] # Изключване на резултатите, които са под прага на допустимата грешка
    results.sort(key=lambda x: x[1], reverse=True) # Сортиране за да може най-самоуверения отговор да се позиционира на първо място
    returnList = []
    for r in results:
        returnList.append((tags[r[0]], r[1]))
    return returnList

def response(sentence):
    results = classify(sentence) # Класификация и търсене на правилен таг
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    return print(random.choice(i['responses'])) # Произволен отговор от съотвения таг
            results.pop(0)

print("Кажи?")
while True: # Създаване на интерактивен цикъл, за разговор между потребителя и чатбота
    userInput = input("Вие: ")
    answer = response(userInput)
    answer
