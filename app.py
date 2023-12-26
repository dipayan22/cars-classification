import os
import sys
import numpy as np



from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras. preprocessing import image



# define flask app
app=Flask(__name__)

MODEL_PATH='artifacts\model_restnet50.h5'

model=load_model(MODEL_PATH)

# function for predict the model
def predict_model(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)

    if preds == 0:
        preds = "The Car is Audi"
    elif preds == 1:
        preds = "The Car is Lamborghini"
    elif preds == 2:
        preds = "The Car is Mercedes"

    return preds


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'notebook/Datasets/uploads')

        # Create the 'uploads' directory if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        pred = predict_model(file_path, model)

        return pred

    return None



if __name__=='__main__':
    app.run(debug=True)

