# CIFAR-10 Image Classifier with ResNet 🚀

This project implements a custom Convolutional Neural Network (CNN) based on the **ResNet architecture** to classify images from the CIFAR-10 dataset. It achieves high accuracy by utilizing Residual Blocks and Global Average Pooling (GAP), effectively addressing the vanishing gradient problem in deep networks.

## 🌟 Key Features
- **ResNet Architecture**: Implemented custom `ResidualBlock` with skip connections to enable deeper network training without performance degradation.
- **Global Average Pooling (GAP)**: Replaced traditional fully connected layers with GAP to minimize parameter count and prevent overfitting.
- **Modular Design**: Refactored monolithic notebook code into modular components (`model.py` for architecture, `inference.py` for execution) for better maintainability and scalability.
- **High Performance**: Achieved **~77% accuracy** on the CIFAR-10 validation set.

## 📂 Project Structure
```bash
├── model.py           # Defines BetterNet architecture & ResidualBlock class
├── inference.py       # Main script for single image inference
├── requirements.txt   # List of Python dependencies
├── better_net.pth     # Trained model weights (Required for inference)
└── README.md          # Project documentation
💻 How to Run
1. Clone the repository
Bash

git clone [https://github.com/shinyoungin4137/AI_Project.git](https://github.com/shinyoungin4137/AI_Project.git)
cd AI_Project
2. Install Dependencies
Bash

pip install -r requirements.txt
3. Prepare Model Weights
⚠️ Important: The trained model weight file (better_net.pth) is required to run the inference.

Since the weight file is excluded from the repository (via .gitignore), you must have the file locally.

Please place your trained better_net.pth file in the root directory of this project.

4. Run Inference
You can test the model with any custom image.

Bash

# Run with default settings (looks for a default image if configured)
python inference.py

# Run with a specific image file path
python inference.py assets/test_dog.jpg
📊 Model Architecture
The network (BetterNet) consists of the following stages:

Stem Layer: Initial 3x3 Convolution, BatchNorm, ReLU, and MaxPool for feature extraction.

Residual Stages:

Stage 1: 32 → 64 channels (Stride 1)

Stage 2: 64 → 128 channels (Stride 2, Downsampling)

Stage 3: 128 → 256 channels (Stride 2, Downsampling)

Classifier Head: Adaptive Average Pooling (GAP) + Linear Layer (FC).

📝 Supported Classes
The model is trained to classify the following 10 categories: plane, car, bird, cat, deer, dog, frog, horse, ship, truck

👤 Author
Name: Youngin Shin

Role: AI & Content Developer

Contact: shinyoungin4137 (via GitHub Issues)
