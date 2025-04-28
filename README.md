# ID Card Classification and OCR

This repository contains Python scripts for classifying ID card images and extracting text using deep learning and OCR techniques. The projects focus on multiclass and binary image classification, along with text extraction from ID card images.

## Projects Overview

1. **ID Card Multiclass Classification**
   - Implements a CNN with a VGG16 backbone to classify ID card images into five categories: Aadhar, Driver's License, PAN, Passport, and Voter ID.
   - Uses `ImageDataGenerator` for data augmentation and preprocessing, with grayscale images resized to 224x224.
   - Trained with early stopping and learning rate reduction for high validation accuracy.

2. **Binary Image Classification**
   - Builds a CNN for binary classification of images from the `HKPFItems` dataset.
   - Features a sequential model with convolutional layers, batch normalization, and dropout, trained on 150x150 RGB images.
   - Includes early stopping and visualizes training/validation accuracy and loss.

3. **OCR for ID Card Text Extraction**
   - Uses PaddleOCR to extract text from Indian PAN card images, with visualization of bounding boxes.
   - Configured for English text detection with DejaVuSans font for visualization.
   - Explored KerasOCR and Tesseract for text recognition.

## Dependencies
- Python 3.x
- TensorFlow, NumPy, Pandas, Matplotlib
- PaddleOCR, PaddlePaddle
- Fonts (e.g., DejaVuSans)

## Instructions
- Run scripts in Google Colab to avoid dependency issues.
- Upload datasets (`IDCardMultiClass.zip`, `HKPFItems.v2i.folder.zip`) to `/content/drive/MyDrive/`.
- Install libraries: `pip install paddleocr paddlepaddle tensorflow`.
- Install fonts: `apt-get install -y fonts-dejavu`.
- Organize datasets in `/content/train`, `/content/valid`, and `/content/test` directories.

## Notes
- Models use callbacks to optimize performance and prevent overfitting.
- PaddleOCR is set to `lang='en'` for English text detection.
- Ensure datasets are preprocessed and correctly pathed before execution.

Explore the scripts for insights into ID card classification and text extraction!
