from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image

import io
import requests
import streamlit as st
import numpy as np
from PIL import Image

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# Векторизованные операции
# Применяются к каждому элементу массива отдельно
x_train = x_train / 255 
x_test = x_test / 255 
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Создаем последовательную моделья
model = Sequential()

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
# Скрытый слой
# Входной полносвязный слой, 800 нейронов, 784 входа в каждый нейрон
model.add(Dense(900, input_dim=784, activation="relu"))

# Выходной полносвязный слой, 10 нейронов (по количеству типов одежды)
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

def preprocess_image(img):
    img = img.resize((28, 28))
    img = img.convert('L')
    #st.image(img)
    x = image.img_to_array(img)
    # Меняем форму массива в плоский вектор
    x = x.reshape(1, 784)
    # Инвертируем изображение
    x = 255 - x
    # Нормализуем изображение
    x /= 255
    return x


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
    
def print_predictions(preds):
    #находит индекс максимального элемента
    index = np.argmax(preds)
    #округляет элемент, который находится на 1 позиции под номером индекс (предс это не массив, а н-мерный массив)
    percent = str(round(preds[0, index] * 100, 4))  
    st.write(preds)
    st.write( "**Номер категории:** " + str(index))
    st.write("**Это** " + str(classes[index]) + " **на** " + percent + " **%** " )

st.title('Распознавание одежды на изображениях')

#крутилки
epoch = st.slider("Выберите количество эпох", 10, 130, 10)

training = st.button('Обучить сеть')

if training:
    #st.write('Обучаем, подождите...')
    history = model.fit(x_train, y_train, 
                    batch_size=200, 
                    epochs = int(epoch),
                    validation_split=0.2,
                    verbose=1)
    model.save('fashion_mnist_dense.h5')
    scores = model.evaluate(x_test, y_test, verbose=1)
    st.balloons()
    st.success("Доля верных ответов на тестовых данных, в процентах: " +  str(round(scores[1] * 100, 4)), icon="✅")
    
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)
