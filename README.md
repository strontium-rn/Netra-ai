# Netra.ai - Eye Disease Detection Project

Netra.ai is a deep learning-based application for detecting eye diseases using transfer learning, deployed via a Streamlit interface. This project is part of our semester work at Kathmandu University.

## Overview
Netra.ai employs transfer learning by combining ResNet50 and ResNet152 models into an ensemble, achieving 86% accuracy. The model is deployed through a Streamlit web application.

## Features
- **Image Upload & Prediction:** Diagnose eye diseases from uploaded images.
- **High Accuracy:** Ensemble model (ResNet50 + ResNet152) with 86% accuracy.
- **Interactive UI:** Built using Streamlit for easy use.

## 📂 Project Structure
```plaintext
Netra.ai/
├── model/
│   └── netra_model.h5        # Trained ensemble model
├── app/
│   └── app.py                # Streamlit app script
├── data/
│   └── sample_images/        # Sample images for testing
└── README.md
```
## Model Development
### Approach
- **Transfer Learning: Combined outputs from pre-trained ResNet50 and ResNet152.
- **Fine-Tuning: Customized both models using the OKDIR-5K eye disease dataset.
- **Performance: Achieved 86% accuracy on the test set.

### Tools and libraries
- **Model Training: TensorFlow, Keras
- **Web Interface: Streamlit
- **Data Handling: NumPy, Pandas, Matplotlib

## Installation and Execution
** Clone Repo:
```plaintext
git clone https://github.com/stronitum-rn/netra.ai.git
cd netra.ai
```
** Launch the application
```
streamlit run app.py
```
