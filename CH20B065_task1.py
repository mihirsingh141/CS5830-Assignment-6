
# Importing the necessary libraries 

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from tensorflow import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

import argparse

import uvicorn

import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title='MNIST Application')

# Defining a function to parse command line arguments

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load model for digit prediction")
    parser.add_argument("model_path", type=str, help="Path to the model file")
    args = parser.parse_args()
    return args.model_path

# Defining a function to load a keras model stored on the local machine

def load_model_from_disk(file_path):
    model = load_model(file_path)

    return model

# Defining a function that takes in a serialized image and returns the predicted digit

def predict_digit(model,data_point):
    prediction = model.predict(data_point.reshape(1,-1))

    return str(np.argmax(prediction))

# Create an API endpoint “@app post(‘/predict’)” that will read the bytes from the uploaded imageto create an serialized array of 784 elements

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Reading the contents of the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_arr = np.array(image.convert('L'))/255 # Converting the 28x28 image to grayscale and normalizing the pixel values

    # Reading the command line argument that stores the path of the model
    path = parse_arguments()
    model = load_model_from_disk(path)
    
    prediction = predict_digit(model, image_arr)

    return {"digit": prediction}

if __name__ == '__main__':
    # Running the web-application defined earlier
    uvicorn.run("CH20B065_task1:app", host="0.0.0.0", port=8000, reload=True)