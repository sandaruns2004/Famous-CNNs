import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms,models

import os
print(os.getcwd())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 2)
model.load_state_dict(torch.load('vgg16/cAT_dOG_cLASSIFICATION/vgg16_cat_dog.pth', map_location=device))
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

def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_image(file_path, model)
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        label_image.config(image=img)
        label_image.image = img
        label_image.config(text=f'Prediction: {prediction}', compound='top')
        

root = tk.Tk()
root.title("Cat vs Dog Classifier")

btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack()

label_image = tk.Label(root)
label_image.pack()

root.mainloop()