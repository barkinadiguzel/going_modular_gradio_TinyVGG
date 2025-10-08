# Going Modular: Pizza, Steak, Sushi Classifier

A modular PyTorch project that trains a TinyVGG model to classify images of pizza, steak, and sushi. This repository includes a Gradio interface for interactive predictions, deployed on Hugging Face Spaces.

## Features
- **Modular Structure**: Organized code with separate modules for data loading (`data_setup.py`), model definition (`model_builder.py`), training/testing (`engine.py`), utilities (`utils.py`), predictions (`predictions.py`), and training script (`train.py`).
- **TinyVGG Model**: A lightweight CNN for classifying pizza, steak, and sushi images.
- **Gradio Interface**: Interactive web app to upload images and get predictions.
- **Hugging Face Spaces**: Live demo hosted online.
  
## Files
- `data_setup.py`: DataLoader creation.
- `model_builder.py`: TinyVGG model definition.
- `engine.py`: Training/testing loops.
- `utils.py`: Model saving/loading.
- `predictions.py`: Prediction functions.
- `train.py`: Model training script.
- `app.py`: Gradio interface.
- `models/05_going_modular_script_mode_tinyvgg_model.pth`: Trained weights.

 ## Requirements
```bash
pip install -r requirements.txt
```
## Feedback
For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
