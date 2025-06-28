# Deep Fake Detection Project
  This folder contains all the code, example images, a final report and a poster designed for a poster session about Deep Learning projects that were developed for the Deep Fake Detection Project.

## Contents

- **Gradcam_examples folder**  
  This folder contains some interpretability elements developed for this project, mainly the gradcam activations in the images.

- **poster.pdf**  
  A poster that was designed and presented for a poster presenting session about Deep Learning and its applications in FCUL's Department of Informatics.

- **report.pdf**  
  This file is the final report that was written for this project, it contains all the results and the methodology that was used when designing the experiments that were performed.

- **train_xception.py**  
  Python script that contains the code developed for the project, this script contains code for an XceptionNet backbone, an EfficientNet-B3 and also different loss functions that were tested.

- **train_xception_lstm.py**  
  This files is exactly the same as the one above, except that it contains an LSTM head in the XceptionNet model's head.

## Usage
To try the project for yourself, you can find the Training, Testing and Validation sets that I curated for this project here:
- **Testing:** https://mega.nz/file/GcVgFRLY#sPSe3DbVd9ThJhdXluvFqttjLrk5Iarq9kKYKCtAboo
- **Training:** https://mega.nz/file/XQ0FHJLT#-6I4eyTgPReiGT8UKdYnxzxODGLjz-ct4ddEjaYr8JI
- **Validation:** https://mega.nz/file/DZ8EiABS#QZ8elzaTwRnK-8a8UW3yHzE3YOcOjAg9sZqM_HVE9sQ

After downloading these files, place them in the same folder as the scripts and you're ready to go.

To run the model without temporal cues:
```bash
python train_xception.py
```
To run the model with temporal cues (LSTM):

```bash
python train_xception_lstm.py
