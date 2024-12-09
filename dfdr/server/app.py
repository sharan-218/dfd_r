# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# app = Flask(__name__)

# # Load your pre-trained model
# model = load_model('./models/dfdc_Xception_ft.h5')

# def preprocess_image(file_path):
#     img = image.load_img(file_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize the image
#     return img_array

# @app.route('/api/detect_deepfake', methods=['POST'])
# def detect_deepfake():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file:
#         filename = secure_filename(file.filename)
#         file_path = os.path.join('uploads', filename)
#         file.save(file_path)
        
#         # Preprocess the image
#         preprocessed_img = preprocess_image(file_path)
        
#         # Make prediction
#         prediction = model.predict(preprocessed_img)
        
#         # Interpret the prediction
#         result = 'real' if prediction[0][0] > 0.5 else 'deepfake'
        
#         # Clean up - delete the uploaded file
#         os.remove(file_path)
        
#         return jsonify({'result': result})

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image 
import io

app = Flask(__name__)
CORS(app)

# Load a pre-trained ResNet model (replace this with your actual deepfake detection model)
model = models.resnet18(pretrained=True)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(file):
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img_t = transform(img)
    return torch.unsqueeze(img_t, 0)

@app.route('/api/detect_deepfake', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        input_tensor = preprocess_image(file)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # For demonstration, we're using the highest confidence class
        # In a real deepfake detector, you'd interpret these results differently
        _, predicted_idx = torch.max(output, 1)
        
        # Simulate deepfake detection result
        result = 'real' if predicted_idx.item() % 2 == 0 else 'deepfake'
        confidence = torch.nn.functional.softmax(output, dim=1)[0, predicted_idx.item()].item()
        
        return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='localhost',port=5000,debug=True)