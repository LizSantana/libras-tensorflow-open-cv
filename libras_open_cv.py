import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

#checkpoint_path = "training_gender/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

model = models.load_model('model_libras2.h5')
#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
video = cv2.VideoCapture(0)



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
        p = np.argmax(prediction, axis=1)
        print(p)
        cv2.putText(frame, str(p), (700, 300), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
        #print(prediction[0][0])
        #Customize this part to your liking...
        #if(prediction == 1 or prediction == 0):
        #    print("No Human")
        #elif(prediction < 0.5 and prediction != 0):
        #    print("Female")
        #elif(prediction > 0.5 and prediction != 1):
        #    print("Male")

        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()
