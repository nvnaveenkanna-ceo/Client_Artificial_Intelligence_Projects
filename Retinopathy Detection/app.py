from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from keras.utils import get_custom_objects
import tensorflow.keras.backend as K

# Define the custom focal loss
def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=0.25):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * K.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return K.sum(loss, axis=1)

# Register it
get_custom_objects().update({'focal_loss_fixed': focal_loss_fixed})

# Load your trained model
model = load_model('model/final_ratinopathy_model.keras')

# Define class labels
classes = ['No DR', 'Mild DR', 'Severe DR']

# Image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Change if your model uses a different size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            prediction = model.predict(img)
            class_index = np.argmax(prediction)
            confidence = round(100 * float(np.max(prediction)), 2)

            return render_template('result.html',
                                   result=classes[class_index],
                                   confidence=confidence,
                                   filename=filename)
    return render_template('index.html')

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 locally
    app.run(host="0.0.0.0", port=port)

