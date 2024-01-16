import json
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

data ={}
number_to_name = {}
model = None

def load_artifacts():
    print('loading saved artifacts...start')
    global model
    if model is None:
        model = load_model('./artifacts/model.h5')


def classify(image_file):
    img = get_croped_image(image_file)
    if img is None:
        return "No Face Detected "
    resized_img = cv2.resize(img,(224,224))
    array_img = image.img_to_array(resized_img)
    final_img = np.expand_dims(array_img, axis=0)
    final_img = preprocess_input(final_img)
    result = model.predict(final_img)[0]
    return result

def get_croped_image( image_file):
    faces_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')
    
    if image_file:
        img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    else :
        return None 
    if img is not None:
        face = faces_cascade.detectMultiScale(img)
        for (x,y,w,h) in face:
            croped_gray = img[y:y+h, x:x+w]
            eyes = eyes_cascade.detectMultiScale(croped_gray)
            if len(eyes) >=2:
                return croped_gray
    return None