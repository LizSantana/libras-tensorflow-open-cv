import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf


with open('classes_libras.txt', 'r') as f:
    classes = f.read().split('\n')

model = models.load_model('model_libras2.h5')

video = cv2.VideoCapture(0)

def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), classes[pred_class]

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into dimensions you used while training
        im = im.resize((64,64))
        img_array = np.array(im)

        #Expand dimensions to match the 4D Tensor shape.
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict function using keras
        prediction = model.predict(img_array)
        p = keras_predict(model, img_array)
        print(p)
        cv2.putText(frame, f'Letra: {str(p[1])}, Precisão: {str(p[0])}', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()
