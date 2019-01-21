
import numpy as np
import cv2
import cnn
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import SGD
from keras.models import model_from_json


model = cnn.model_cnn()
#model = model.load_weights('model.h5')
#sgd = SGD(lr=0.001)
# load json and create model
json_file = open('model_sgd.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_sgd.h5")
print("Loaded model from disk")
print(loaded_model)
loaded_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
#model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])
 


while True:
    path = input('Image path:')
    img = cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(28,28))
    print(gray.shape[0])
    cv2.imshow("image",gray)
    
    binary_image = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow("image1",binary_image)
    print("binary image")
    binary_image = np.reshape(binary_image,[1,1,28,28])
    print("img")
    print(binary_image.shape)
    classes = loaded_model.predict_classes(binary_image)
    print("'0' represent as even class and '1' represent as odd clas")
    print("Output Class:",classes)
    print("If you wanna continue press any key except 'q'")
    k = cv2.waitKey()
    if k == ord('q'):
        cv2.destroyAllWindows()
        break;
    
    
print("exit")
    
        

#img = cv2.imread('test.jpg')
#img = cv2.resize(img,(320,240))
#img = np.reshape(img,[1,320,240,3])

#classes = model.predict_classes(img)