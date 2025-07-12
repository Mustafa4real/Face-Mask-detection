# Face-Mask-detection
In this project I used **Teachable Machine** to train a Machine learning model to recognize if a person in image wears a mask normally, wears it incorrectly or doesn't wear at all.

<img width="1890" height="862" alt="wearing mask" src="https://github.com/user-attachments/assets/2dd47407-d9ce-48c5-a53b-4f0dde1c7356" />

## Dataset
Source: https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection?resource=download

Contains 3 classes: `Wearing`, `Not wearing`, `Wearing incorrectly`.

## Pipeline
- 500 images of each class were used for the training in **Teachable Machine**, then model was exported and saved as `trained_model.h5`.
- Python script "`face_mask_detector.py`" was written to make a program that loads the model, accept input image and predict its class.
