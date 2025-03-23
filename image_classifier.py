# وارد کردن کتابخانه‌ها
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Model training
def train_model():
    # Loading data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Data normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Model making
    model = Sequential([
        layers.Flatten(input_shape=(28, 28)),  
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')  
    ])

    # Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Training
    model.fit(x_train, y_train, epochs=5)

    # Evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    # save
    model.save('mnist_model.h5')

# Image loading and preview
def predict_image(image_path):

    model = keras.models.load_model('mnist_model.h5')

    img = Image.open(image_path)

    img = img.convert('L')  
    img = img.resize((28, 28))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    print(f'Predicted Class: {predicted_class}')

# Getting input from the user
def get_image_path():

    root = tk.Tk()
    root.withdraw()  

    file_path = filedialog.askopenfilename(
        title="Select an image to classify", 
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.jfif")]
    )
    
    return file_path

# Run
def main():
    print("Please select an image to classify (in .jpg, .png, or .jfif format):")
    
    image_path = get_image_path()  
    if image_path:
        predict_image(image_path)
    else:
        print("No image selected.")


train_model()


main()
