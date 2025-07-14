# Fashion MNIST Web App

This is a simple Flask web application that uses a trained CNN model on Fashion MNIST dataset to predict clothing categories.

## Features
- Upload an image of clothing (28x28 grayscale).
- Model predicts one of the 10 classes (T-shirt/top, Trouser, Pullover, etc).

## How to Run
1. Clone this repo and navigate to the folder.
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Start the Flask server:
    ```
    python app.py
    ```
4. Open your browser and go to `http://127.0.0.1:5000/`.

## Files
- `app.py`: Flask backend
- `Fashion_mnist.hdf5`: Pre-trained model
- `templates/index.html`: Upload page
- `templates/result.html`: Result page

## Author
- ❤️ Built by Balachandharsriram.M as a beginner full stack project.
