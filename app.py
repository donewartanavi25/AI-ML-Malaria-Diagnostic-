import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
# Note: Ensure you have run train_model.py first!
MODEL_PATH = 'malaria_model.h5'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    print("WARNING: Model not found. Please run train_model.py first.")

def predict_image(img_path):
    # Load and resize the image to match training data
    img = image.load_img(img_path, target_size=(64, 64))
    # Convert image to numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the batch format the model expects: (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict (Output is a probability between 0 and 1)
    prediction = model.predict(img_array)
    
    # Alphabetical classes: 0 = parasitized, 1 = uninfected
    if prediction[0][0] > 0.5:
        return "Uninfected"
    else:
        return "Parasitized"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected.")
        
        if file and model:
            # Save the file securely
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Run prediction
            result = predict_image(filepath)
            
            # Pass result and image path to HTML
            return render_template('index.html', prediction=result, image_path=filepath)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
