# server.py
# Stable version

import os
import sys

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import io

# ---------------- Path setup ----------------
# Resolve every path relative to this file's location, not the current
# working directory, so it doesn't matter where you launch uvicorn from.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")

for path, label in [(STATIC_DIR, "static"), (TEMPLATES_DIR, "templates")]:
    if not os.path.isdir(path):
        sys.exit(
            f"Startup failed: '{label}' folder not found at {path}\n"
            f"Make sure a '{label}' folder exists next to server.py."
        )

if not os.path.isdir(MODEL_DIR):
    sys.exit(
        f"Startup failed: trained_model folder not found at {MODEL_DIR}\n"
        f"Run train.py first (it saves the model to 'trained_model'), "
        f"or make sure 'trained_model' sits next to server.py."
    )

if not os.path.isfile(os.path.join(TEMPLATES_DIR, "index.html")):
    sys.exit(
        f"Startup failed: templates/index.html not found in {TEMPLATES_DIR}"
    )

# ---------------- App setup ----------------

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------------- Load model ----------------

try:
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    model.eval()
except Exception as e:
    sys.exit(f"Startup failed: could not load model from {MODEL_DIR}\n{e}")

# Explanations shown in UI
EXPLANATIONS = {
    "healthy": "The fish appears healthy. Continue monitoring water quality and behavior.",
    "parasitic": "The fish may show signs of a parasitic infection (e.g. White Spot/Ich). Consider checking water parameters and consulting an aquatic specialist about treatment options.",
    "fungal": "The fish may show signs of a fungal infection. Fungal infections often follow injury or poor water quality - check water parameters and consider consulting an aquatic specialist.",
    "bacterial": "The fish may show signs of a bacterial infection (e.g. Fin Rot). Consider checking water parameters and consulting an aquatic specialist about treatment options.",
    "dropsy": "The fish may show signs of dropsy (swelling, raised scales). This can indicate a serious underlying issue - consult an aquatic specialist promptly."
}

# ---------------- Routes ----------------

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    confidence = float(torch.max(probs)) * 100
    pred_index = int(torch.argmax(probs))

    label = model.config.id2label[pred_index].lower()

    return JSONResponse({
        "prediction": label,
        "confidence": round(confidence, 2),
        "explanation": EXPLANATIONS.get(label, "")
    })


# ---------------- Entrypoint ----------------
# Lets you run this with `python server.py`. You can still also run it with
# `uvicorn server:app --reload` from the command line if you prefer.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)