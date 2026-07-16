import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Class mapping - now 5 classes instead of 2.
# IMPORTANT: these labels must exactly match your dataset folder names
# (dataset/train/healthy, dataset/train/parasitic, etc.) - ImageFolder
# assigns label indices alphabetically, so double-check this matches
# what ImageFolder actually produces (see the printed class order in
# test_model.py output) rather than assuming this order is correct.
id2label = {
    0: "bacterial",
    1: "dropsy",
    2: "fungal",
    3: "healthy",
    4: "parasitic",
}
label2id = {v: k for k, v in id2label.items()}

# Load pretrained ViT
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=5,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Dataset wrapper
class FishDataset(Dataset):
    def __init__(self, folder_path):
        self.dataset = ImageFolder(folder_path)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label)
        return inputs

# Load datasets
train_dataset = FishDataset("dataset/train")
val_dataset = FishDataset("dataset/val")

# Sanity check: make sure ImageFolder's class order actually matches
# id2label above. If this ever prints something different, fix id2label
# to match rather than the other way around.
print("ImageFolder class order:", train_dataset.dataset.classes)

# Evaluation metrics - switched from binary to multiclass averaging,
# since precision_recall_fscore_support needs to know how to combine
# scores across more than 2 classes now.
def compute_metrics(pred):
    predictions = pred.predictions.argmax(-1)
    accuracy = accuracy_score(pred.label_ids, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        pred.label_ids, predictions, average="weighted", zero_division=0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    load_best_model_at_end=True,
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
model.save_pretrained("trained_model")
processor.save_pretrained("trained_model")