# server.py
# Stable version

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import io

# ---------------- App setup ----------------

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------- Load model ----------------

processor = ViTImageProcessor.from_pretrained("trained_model")
model = ViTForImageClassification.from_pretrained("trained_model")
model.eval()

# Explanations shown in UI
EXPLANATIONS = {
    "healthy": "The fish appears healthy. Continue monitoring water quality and behavior.",
    "unhealthy": "The fish may show signs of illness. Consider checking water parameters or consulting an aquatic specialist."
}

# ---------------- Routes ----------------

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)