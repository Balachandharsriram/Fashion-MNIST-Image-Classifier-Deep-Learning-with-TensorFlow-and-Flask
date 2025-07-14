from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained Fashion MNIST model
model = load_model('Fashion_mnist.hdf5')

# Class labels
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/')
def home():
    # Render the upload page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is present
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Process the image
    try:
        img = Image.open(file.stream).convert('L').resize((28,28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1,28,28,1)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = classes[predicted_class]

        # Return result page
        return render_template('result.html', prediction=predicted_label)
    
    except Exception as e:
        return f"Error processing image: {e}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
