import joblib
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Define the model architecture (ResNet9 from your example)
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))

    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

def load_model(model_path):
    """Load the saved model from the specified path.

    Args:
        model_path (str): Path to the model file.

    Returns:
        model: Loaded model.
    """
    try:
        if model_path.endswith('.pkl') or model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        elif model_path.endswith('.pth') or model_path.endswith('.pt'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Instantiate the model
            num_classes = 38  # Update this to the actual number of classes in your dataset
            model = ResNet9(3, num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        else:
            raise ValueError(f"Unsupported model file format: {model_path}")
        return model
    except Exception as e:
        raise IOError(f"Error loading the model from {model_path}: {e}")

def predict_fertilizer(nitrogen, phosphorous, potassium, model):
    """Predict the type of fertilizer required based on soil composition.

    Args:
        nitrogen (int): Nitrogen content in the soil.
        phosphorous (int): Phosphorous content in the soil.
        potassium (int): Potassium content in the soil.
        model: Trained prediction model.

    Returns:
        tuple: (Recommended fertilizer type, confidence percentage)
    """
    try:
        classes = {0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'}
        X = np.array([[nitrogen, phosphorous, potassium]])
        prediction = model.predict(X)
        probabilities = model.predict_proba(X)
        confidence = np.max(probabilities) * 100
        return classes[prediction[0]], confidence
    except Exception as e:
        raise ValueError(f"Error predicting fertilizer: {e}")

def predict_crop(N, P, K, temperature, humidity, ph, rainfall, model):
    """Predict the recommended crop based on soil and weather conditions.

    Args:
        N (int): Nitrogen ratio in the soil.
        P (int): Phosphorus ratio in the soil.
        K (int): Potassium ratio in the soil.
        temperature (float): Average temperature in degree Celsius.
        humidity (float): Relative humidity in percentage.
        ph (float): pH value of the soil.
        rainfall (float): Yearly rainfall in mm.
        model: Trained prediction model.

    Returns:
        tuple: (Recommended crop, confidence percentage)
    """
    try:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        predicted_class_index = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)
        confidence = np.max(probabilities) * 100
        class_mapping = {0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'}
        crop = class_mapping[predicted_class_index]
        return crop.capitalize(), confidence
    except Exception as e:
        raise ValueError(f"Error predicting crop: {e}")

def predict_crop_disease(image_path, model, device='cpu'):
    """Predict the disease of a crop from an image.

    Args:
        image_path (str): Path to the image file.
        model: Trained prediction model.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        tuple: (Predicted disease category, confidence percentage)
    """
    try:
        image = Image.open(image_path).resize((256, 256))
        image = transforms.ToTensor()(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            confidence = confidence.item() * 100
            predicted_class = predicted.item()

        classes = [
            'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
            'Blueberry - Healthy', 'Cherry (including sour) - Powdery Mildew', 'Cherry (including sour) - Healthy',
            'Corn (maize) - Cercospora Leaf Spot, Gray Leaf Spot', 'Corn (maize) - Common Rust',
            'Corn (maize) - Northern Leaf Blight', 'Corn (maize) - Healthy', 'Grape - Black Rot',
            'Grape - Esca (Black Measles)', 'Grape - Leaf Blight (Isariopsis Leaf Spot)', 'Grape - Healthy',
            'Orange - Haunglongbing (Citrus Greening)', 'Peach - Bacterial Spot', 'Peach - Healthy',
            'Pepper, bell - Bacterial Spot', 'Pepper, bell - Healthy', 'Potato - Early Blight', 'Potato - Late Blight',
            'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew',
            'Strawberry - Leaf Scorch', 'Strawberry - Healthy', 'Tomato - Bacterial Spot', 'Tomato - Early Blight',
            'Tomato - Late Blight', 'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites, Two-spotted Spider Mite',
            'Tomato - Target Spot', 'Tomato - Tomato Yellow Leaf Curl Virus', 'Tomato - Tomato Mosaic Virus', 'Tomato - Healthy'
        ]
        class_name = classes[predicted_class]
        return class_name, confidence
    except Exception as e:
        raise ValueError(f"Error predicting crop disease: {e}")

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
