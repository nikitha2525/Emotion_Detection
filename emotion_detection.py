import os
from flask import Flask, jsonify, request, render_template, url_for
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 16 * 16, 512) # Corrected from 64 * 4 * 4
    self.fc2 = nn.Linear(512 , 5)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = self.pool(self.relu(self.conv3(x)))
    x = x.view(x.size(0), -1)# Corrected from 64 * 4 * 4
    x = self.dropout(x)
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x

device = torch.device("cpu")
model = CNN()
model.load_state_dict(torch.load("Emotion.pth", map_location=device))
model.eval()

transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    print(image.size)
    image = transform_pipeline(image)
    print(image.size)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    classes = {
        0:"Angry",
        1:"Fear",
        2:"Happy",
        3:"Sad",
        4:"Surprise"
    }
    emotion = classes[prediction.item()]
    confidence = f"{confidence.item()*100:.2f}%"
    return emotion, confidence


@app.route("/", methods=["GET","POST"])
def detection():
    if request.method=="POST":
        file=request.files["file"]
        os.makedirs("static",exist_ok=True)
        imgpath=os.path.join("static",file.filename)
        file.save(imgpath)
        emotion, confidence = classify_emotion(imgpath)
        return jsonify({
            "Classification_Result": emotion,
            "Classification_Confidence": confidence
        })
    return render_template("detection.html")


if __name__ =='__main__':
    app.run(debug=True)
    