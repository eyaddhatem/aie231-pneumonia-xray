"""Demo: upload a chest X-ray, get P(pneumonia). Run: pip install gradio && python app/app.py
Screening aid for a course project, NOT a medical device."""
import sys
sys.path.insert(0, ".")
import gradio as gr
from src.predict import load_model, DEFAULT_THRESHOLD, load_image

model = load_model("models/vgg16_final.weights.h5")


def run(img_path, threshold):
    p = float(model.predict(load_image(img_path), verbose=0).ravel()[0])
    return {"PNEUMONIA": p, "NORMAL": 1 - p}, ("PNEUMONIA (flag for a doctor)" if p >= threshold else "NORMAL")


gr.Interface(run, [gr.Image(type="filepath"), gr.Slider(0.05, 0.95, DEFAULT_THRESHOLD, label="threshold")],
             [gr.Label(), gr.Textbox(label="decision")], title="Pediatric chest X-ray pneumonia screening (course demo)").launch()
