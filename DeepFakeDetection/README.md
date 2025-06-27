# Project Overview

This repository contains materials related to a machine learning project involving deep learning models for classification or sequence analysis tasks. It includes training scripts and documentation in the form of a report and a poster.

## Contents

- **poster.pdf**  
  A high-level summary poster of the project, suitable for presentations or conferences. It includes visuals and concise explanations of the approach, results, and key findings.

- **report.pdf**  
  A detailed technical report that describes the problem, methodology, experiments, results, and conclusions. Ideal for readers who want a deeper understanding of the project.

- **train_xception.py**  
  Python script that implements a training pipeline using the Xception model architecture. This is likely used for image classification tasks and serves as the baseline or main model.

- **train_xception_lstm.py**  
  Python script that combines the Xception model with an LSTM (Long Short-Term Memory) layer, possibly for tasks involving temporal sequences of images (e.g., video classification or medical imaging sequences).

## Usage
To try the project for yourself, you can find the Training, Testing and Validation sets that I curated for this project here:
- **Testing:** https://mega.nz/file/GcVgFRLY#sPSe3DbVd9ThJhdXluvFqttjLrk5Iarq9kKYKCtAboo
- **Training:** https://mega.nz/file/XQ0FHJLT#-6I4eyTgPReiGT8UKdYnxzxODGLjz-ct4ddEjaYr8JI
- **Validation:** https://mega.nz/file/DZ8EiABS#QZ8elzaTwRnK-8a8UW3yHzE3YOcOjAg9sZqM_HVE9sQ

To run the model without temporal cues:
```bash
python train_xception.py
```
To run the model with temporal cues (LSTM):

```bash
python train_xception_lstm.py
