Famous CNN Implementations and Applications
===========================================

This repository contains **from-scratch implementations** of famous Convolutional Neural Networks (CNNs) in Python, along with practical applications such as image classification and GUI/web interfaces. It uses **PyTorch** and **TensorFlow** for different implementations.

🚀 Implemented CNN Models
-------------------------

### 1\. **LeNet-5**

*   First CNN architecture (1998).
    
*   Implemented from scratch using **PyTorch**.
    
*   Suitable for handwritten digit classification (e.g., MNIST).
    

### 2\. **AlexNet**

*   Introduced ReLU activations and Dropout.
    
*   Large-scale CNN suitable for ImageNet.
    
*   Implemented from scratch in **PyTorch**.
    

### 3\. **VGG16**

*   Deep but simple architecture with 16 layers.
    
*   Implemented from scratch in **PyTorch**.
    
*   Used for **Cat vs Dog classification** and GUI/web apps.
    

### 4\. **ResNet18**

*   Introduces residual connections to enable very deep networks.
    
*   Implemented from scratch in **PyTorch**.
    

### 5\. **ResNet50 (Flower Classifier)**

*   Fine-tuned model to classify 5 flower types: **Rose, Tulip, Sunflower, Dandelion, Daisy**.
    
*   Uses **PyTorch** for implementation and training.
    

🐱 Cat vs Dog Classification
----------------------------

*   **Model:** VGG16 (custom trained)
    
*   **Applications:**
    
    1.  **Tkinter GUI App** (gui\_app.py):
        
        *   Load an image from your system.
            
        *   Predicts **Cat** or **Dog**.
            
    2.  **Flask Web App** (web\_app/):
        
        *   Upload image through a web interface.
            
        *   Display prediction on the browser.
            

🌻 Flower Classification (ResNet50)
-----------------------------------

*   **Flower Types:** Rose, Tulip, Sunflower, Dandelion, Daisy
    
*   **Implementation:** Fine-tuned ResNet50 using **PyTorch**.
    
*   **Files:**
    
    *   train.py — Training script
        
    *   predict.py — Test single images
        
    *   resnet50\_flower.pth — Trained weights
        

🛠️ Utilities
-------------

*   **GPU Check:** check\_gpu.py
    
    *   Checks CUDA availability and prints GPU info.
        
    *   Simple tensor multiplication test to verify performance.
        
*   **Dataset Loader:** data\_loader.py
    
    *   Standardizes data preprocessing for PyTorch models.
        


## ⚡ Usage Examples

1. **Train Cat vs Dog VGG16:**

```bash
python applications/cat_dog_classification/train.py
```

**Run Tkinter GUI:**

```bash
python applications/cat_dog_classification/gui_app.py
```

**Run Flask Web App:**

```bash
cd applications/cat_dog_classification/web_app
python app.py
```

**Test ResNet50 Flower Classifier:**

```bash
python applications/resnet50/predict.py --image path/to/flower.jpg
```

**Check GPU:**

```bash
python utils/check_gpu.py
```

You can **directly copy-paste** this into your `README.md` and it will render correctly on GitHub.
  
If you want, I can also format the **Installation** and **Requirements** sections in the same style so your README looks fully polished. Do you want me to do that?
🧪 Requirements
---------------

*   Python >= 3.8
    
*   PyTorch >= 2.0
    
*   TensorFlow >= 2.13
    
*   Torchvision, Pillow, Flask, Tkinter (built-in for Python)
    
*   Numpy, Matplotlib, tqdm
    

📖 References
-------------

1.  **LeNet-5** – Yann LeCun, 1998
    
2.  **AlexNet** – Krizhevsky et al., 2012
    
3.  **VGG16** – Simonyan & Zisserman, 2014
    
4.  **ResNet** – He et al., 2015
    
5.  **PyTorch Documentation** – [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
    
6.  **TensorFlow Documentation** – [https://www.tensorflow.org/api\_docs](https://www.tensorflow.org/api_docs)
