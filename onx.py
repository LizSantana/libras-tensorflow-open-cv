
import numpy as np
import os
import cv2

imageDir = 'asl_alphabet_test/asl_alphabet_test'
net =  cv2.dnn.readNetFromONNX('sl.onnx') 
with open('classes.txt', 'r') as f:
    classes = f.read().split('\n')
for i, image_name in enumerate(os.listdir(imageDir)):
    image = cv2.imread(os.path.join(imageDir, image_name))
    blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (224, 224),(0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)
    preds = net.forward()
    biggest_pred_index = np.array(preds)[0].argmax()
    p = classes[biggest_pred_index]
    print(p, image_name)

    
    
    #ax = plt.subplot(6, 5, i + 1)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
   # plt.title("predicted: {}, True: {}".format(classes[biggest_pred_index], image_name.split('_test.jpg')[0]))
    #plt.axis("off")


import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

#checkpoint_path = "training_gender/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
video = cv2.VideoCapture(0)



while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
        im = im.resize((256,256))
        im = np.array(im)

        blob = cv2.dnn.blobFromImage(im, 1.0 / 255, (224, 224),(0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)
        preds = net.forward()
        biggest_pred_index = np.array(preds)[0].argmax()
        p = classes[biggest_pred_index]
        print(p)
        cv2.putText(frame, str(p), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
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
