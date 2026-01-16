from flask import Flask, request, render_template_string
from PIL import Image
import torch
from torchvision import models, transforms
import os

app = Flask(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 2)
model_path = r"C:\Users\ADMIN\Desktop\Famous-CNNs\VGG16\Cat_Dog_Classification\vgg16_cat_dog.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

classes = ['Cat', 'Dog']

# Predict Function
def predict_image(path):
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]

# Flask Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    img_url = ""
    if request.method == "POST":
        file = request.files["file"]
        if file:
            upload_folder = os.path.join("static", "uploads")
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            result = predict_image(file_path)
            img_url = "/static/uploads/" + file.filename

    # HTML with CSS
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cat vs Dog Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f0f2f5; text-align: center; padding: 50px; }
            h1 { color: #333; }
            .upload-section { background: #fff; padding: 20px; border-radius: 10px; display: inline-block; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
            input[type=file] { padding: 10px; margin: 10px 0; }
            input[type=submit] { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            input[type=submit]:hover { background: #45a049; }
            .result { margin-top: 20px; font-size: 24px; font-weight: bold; color: #333; }
            img { margin-top: 20px; max-width: 300px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.2); }
        </style>
    </head>
    <body>
        <h1>Cat vs Dog Classifier</h1>
        <div class="upload-section">
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="file" required onchange="this.form.submit()">
            </form>
        </div>
        {% if result %}
            <div class="result">Prediction: {{ result }}</div>
            <img src="{{ img_url }}" alt="Uploaded Image" style="max-width:300px; border-radius:10px;">

        {% endif %}
    </body>
    </html>
    """, result=result, img_url=img_url)

if __name__ == "__main__":
    app.run(debug=True)
