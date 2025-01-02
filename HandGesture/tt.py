from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os


model_path = 'D:/AI-ML course/AI Programs/HandGesture/model.h5'

if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model Loaded Successfully")
else:
    print(f"Error: The model file '{model_path}' does not exist.")
    exit()

def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size=(256, 256), grayscale=True)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    arr = np.array(result[0])
    print(arr)
    maxx = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1
    classes = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
    result = classes[max_prob - 1]
    print(img_name, result)


path = 'D:/MasterClass/Artificial_Intelligence/Day13/Dataset/val/TWO'
files = []


for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))

# Classify each image file
for f in files:
    classify(f)
    print('\n')
