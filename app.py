# app.py
import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from model_builder import TinyVGG
from timeit import default_timer as timer

# Setup device and class names
device = "cuda" if torch.cuda.is_available() else "cpu"
class_names = ["pizza", "steak", "sushi"]

# Load model
model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names))
model.load_state_dict(torch.load("models/05_going_modular_script_mode_tinyvgg_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform (matches train.py)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Prediction function
def predict(img) -> tuple[dict, float]:
    start_time = timer()
    try:
        img = transform(Image.fromarray(img).convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            pred_probs = torch.softmax(model(img), dim=1)
        pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
        pred_time = round(timer() - start_time, 4)
        return pred_labels_and_probs, pred_time
    except Exception as e:
        return {"Error": str(e)}, 0.0

# Gradio interface
title = "Pizza, Steak, Sushi Classifier"
description = "A TinyVGG model to classify images as pizza, steak, or sushi."
article = "Built with PyTorch and Gradio."
example_list = [["examples/" + example] for example in os.listdir("examples") if example.endswith((".jpg", ".png"))]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[gr.Label(num_top_classes=3, label="Predictions"), gr.Number(label="Prediction time (s)")],
    examples=example_list,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(debug=False)
