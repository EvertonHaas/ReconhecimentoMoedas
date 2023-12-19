import cv2
import numpy as np
from keras.models import load_model
from config import model_path, classes, data_shape

# carregegar o modelo
model = load_model(model_path, compile=False)
data = np.ndarray(shape=data_shape, dtype=np.float32)


def pre_process(img):
    img_pre = cv2.GaussianBlur(img, (9, 9), 3)
    img_pre = cv2.Canny(img_pre, 90, 140)
    kernel = np.ones((4, 4), np.uint8)
    img_pre = cv2.dilate(img_pre, kernel, iterations=2)
    img_pre = cv2.erode(img_pre, kernel, iterations=1)
    return img_pre


def detectar_moeda(img):
    img_moeda = cv2.resize(img, (224, 224))
    img_moeda = np.asarray(img_moeda)
    img_moeda_normalize = (img_moeda.astype(np.float32) / 127.0) - 1

    cv2.imshow('IMG PRE', img_moeda_normalize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    data[0] = img_moeda_normalize
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = classes[index]
    return classe, percent
