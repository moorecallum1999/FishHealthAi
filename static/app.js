// static/app.js
// Simple stable version — upload -> preview -> prediction

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

const preview = document.getElementById("preview");
const previewPlaceholder = document.getElementById("previewPlaceholder");

const predictionBadge = document.getElementById("predictionBadge");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const explanationText = document.getElementById("explanationText");

// ---------- UI helpers ----------

// All possible prediction labels the badge needs to style differently.
// Keep this in sync with server.py's EXPLANATIONS keys / your model's classes.
const RESULT_CLASSES = ["healthy", "parasitic", "fungal", "bacterial", "dropsy"];

function setAnalyzing() {
  predictionText.textContent = "Analyzing image...";
  confidenceText.textContent = "--%";
  explanationText.textContent = "";
  confidenceBar.style.width = "0%";
  predictionBadge.classList.remove(...RESULT_CLASSES);
}

function setResult(prediction, confidence, explanation) {

  predictionBadge.classList.remove(...RESULT_CLASSES);

  if (RESULT_CLASSES.includes(prediction)) {
    predictionBadge.classList.add(prediction);
  }

  predictionText.textContent = `Predicted: ${prediction}`;
  confidenceText.textContent = `${confidence}%`;
  confidenceBar.style.width = `${confidence}%`;
  explanationText.textContent = explanation;
}

function showPreview(file) {
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
  previewPlaceholder.style.display = "none";
}

// ---------- Upload handling ----------

async function analyzeImage(file) {

  setAnalyzing();
  showPreview(file);

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/predict", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  setResult(
    data.prediction,
    data.confidence,
    data.explanation
  );
}

// Drag & drop
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");

  const file = e.dataTransfer.files[0];
  if (file) analyzeImage(file);
});

// Click upload
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) analyzeImage(file);
});