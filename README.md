# EmotionSense AI - Facial Emotion Detection

Detects and classifies human emotions from facial expressions using a CNN trained on FER-2013 and OpenCV face detection.

## Features
- CNN classifier for 7 emotions
- Haar cascade face detection
- Single-image prediction + real-time webcam prediction
- Data augmentation and early stopping during training

## Setup
pip install -r requirements.txt

## Train
python train.py --csv data/fer2013.csv --epochs 30 --out emotionsense.h5

## Predict
python predict_image.py --model emotionsense.h5 --image face.jpg
python predict_webcam.py --model emotionsense.h5

## Stack
Python, TensorFlow/Keras, OpenCV, NumPy, pandas
