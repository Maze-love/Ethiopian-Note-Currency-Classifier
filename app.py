import os
import sys
import uuid
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from gtts import gTTS

# 1. Prevent Windows network/proxy interception on localhost
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

# 2. Application Constants
MODEL_PATH = "ethiopian_currency_model.keras"
IMG_SIZE = 100  # Target dimensions (100x100) from notebook training
CLASS_NAMES = ["አምስት ብር", "አስር ብር", "ሃምሳ ብር", "መቶ ብር", "ሁለት መቶ ብር"]

model = None


def load_keras_model():
    """Lazily loads and verifies the TensorFlow model upon first request."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' was not found in the root directory!"
            )
        print(f"Loading TensorFlow model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model successfully loaded into memory.")
    return model


def predict_currency_with_voice(input_image):
    """Pipeline for image preprocessing, prediction, and voice output generation."""
    if input_image is None:
        return {}, None

    try:
        # Load model instance
        net = load_keras_model()

        # Image Preprocessing: PIL conversion -> resize -> normalization
        img = Image.fromarray(input_image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension: shape (1, 100, 100, 3)
        img_tensor = np.expand_dims(img_array, axis=0)

        # Execute inference
        predictions = net.predict(img_tensor, verbose=0)[0]

        # Format predictions into dictionary for Gradio
        pred_dict = {
            CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
        }

        # Determine top predicted class
        top_class_index = int(np.argmax(predictions))
        top_class = CLASS_NAMES[top_class_index]

       # Generate unique temporary audio file safely on Windows
        audio_filename = f"speech_{uuid.uuid4().hex}.mp3"
        speech_text = f"{top_class}"
        
        tts = gTTS(text=speech_text, lang="am")
        tts.save(audio_filename)

       
        
        return pred_dict, audio_filename

    except Exception as e:
        print(f"Prediction Pipeline Error: {e}", file=sys.stderr)
        return {"Inference Error": 0.0}, None


# ---------------------------------------------------------
# Gradio Interface Definition
# ---------------------------------------------------------
title = "Ethiopian Banknote Currency Classifier"
description = (
    "Upload an image of an Ethiopian banknote (5, 10, 50, 100, or 200 Birr) "
    "to identify its denomination and hear the predicted voice output."
)

demo = gr.Interface(
    fn=predict_currency_with_voice,
    inputs=gr.Image(label="Upload Banknote Image", type="numpy"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predicted Denomination"),
        gr.Audio(label="Voice Announcement", autoplay=True),
    ],
    title=title,
    description=description,
    theme="soft",
)

# ---------------------------------------------------------
# Web Server Launch Configuration
# ---------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        prevent_thread_lock=False,
    )