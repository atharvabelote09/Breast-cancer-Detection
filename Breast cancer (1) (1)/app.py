from flask import Flask, render_template, request, current_app
from werkzeug.utils import secure_filename
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import os 

import label_image  # your model prediction script
import image_fuzzy_clustering as fem  # clustering script

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route: Home
@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

# Route: Login page
@app.route('/login')
def login():
    return render_template('login.html')

# Route: Chart display
@app.route('/chart')
def chart():
    return render_template('chart.html')

# Route: Upload page
@app.route('/upload')
def upload():
    return render_template('index1.html')

# Route: Success after clustering
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        cluster_num = request.form.get('cluster')
        file = request.files['file']

        filename = secure_filename(file.filename)
        saved_path = save_img(file, filename)

        # Apply fuzzy clustering algorithm
        fem.plot_cluster_img(saved_path, cluster_num)

        return render_template('success.html')

# Save image to disk
def save_img(img, filename):
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(img)
    image.save(save_path)
    return save_path

# Route: Index (model prediction)
@app.route('/index')
def index():
    return render_template('index.html')

# Load and predict the image
def load_image(image_path):
    result = label_image.main(image_path)
    return result

# Route: Predict disease stage
@app.route('/predict', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict with model
        result = load_image(file_path).title()

        # Disease stage explanations
        stage_info = {
            "0": " → There is no Chance of Breast Cancer, No Risk. You are Healthy.",
            "1": " → Stage1 - The disease is only in the ducts and lobules of the breast. It has not spread to the surrounding tissue. It is also called noninvasive cancer (Tis, N0, M0).",
            "2": " → Stage2 - Invasive cancer. Tumor may not be in the breast, but cancer cells have spread to 1-3 lymph nodes. May show a 2 to 5 cm tumor with or without spread.",
            "3": " → Stage3 - Cancer cells have spread to at least 1 to 3 lymph nodes. May show a 2 to 5 cm tumor in the breast with or without spread.",
            "4": " → Stage4 - Tumor of any size with spread to lymph nodes. The disease has spread to more than 4 lymph nodes in the breast or axilla."
        }

        full_result = result + stage_info.get(result, " → Unknown Stage Detected")

        # Remove saved image
        os.remove(file_path)

        return render_template('result.html', prediction=full_result)

    return render_template('index.html')

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
