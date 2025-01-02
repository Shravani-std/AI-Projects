from keras.preprocessing import image
from keras.models import load_model
import numpy as np

model_path = 'D:\AI-ML course\AI Programs\HandGesture\model.h5'
model = load_model(model_path)
print("Model Loaded Successfully")

def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size = (256, 256),grayscale=True)

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    arr = np.array(result[0])
    print(arr)
    maxx = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1
    classes=["None", "One", "Two", "Three", "Four", "Five"]
    result = classes[max_prob - 1]
    print(img_name,result)


import os
path = 'D:\AI-ML course\AI Programs\HandGesture\test'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.png' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')
