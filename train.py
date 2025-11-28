import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import timm
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =============================
# Custom Dataset
# =============================
class LungCancerDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.paths = df['file_path'].values
        self.labels = df['label_id'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# =============================
# Training & Validation Functions
# =============================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return running_loss/len(loader), acc, f1

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return running_loss/len(loader), acc, f1

@torch.no_grad()
def get_all_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    model.to(device)
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)

# =============================
# Main function
# =============================
def main():
    # -----------------------------
    # Configurations
    # -----------------------------
    BASE_PATH = r"D:\lung_cancer_project\lung_cancer_detection\data"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = r"./models/saved_models"
    os.makedirs(output_dir, exist_ok=True)  # Ensure folder exists
    model_name = "efficientnet_b3"
    epochs = 20
    lr = 1e-4
    weight_decay = 1e-5

    # -----------------------------
    # Load Data
    # -----------------------------
    dfs = {}
    for split in ['train', 'val', 'test']:
        image_paths = glob(os.path.join(BASE_PATH, split, '*/*.jpg'))
        df_split = pd.DataFrame(image_paths, columns=['file_path'])
        df_split['label'] = df_split['file_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
        dfs[split] = df_split

    class_names = sorted(pd.concat(dfs.values())['label'].unique())
    num_classes = len(class_names)
    label2id = {name: i for i, name in enumerate(class_names)}
    id2label = {i: name for i, name in enumerate(class_names)}

    for split in dfs:
        dfs[split]['label_id'] = dfs[split]['label'].map(label2id)

    print(f"Classes: {class_names}")
    print(f"Training samples: {len(dfs['train'])}, Validation samples: {len(dfs['val'])}, Test samples: {len(dfs['test'])}")

    # -----------------------------
    # Transforms
    # -----------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # -----------------------------
    # Datasets and Dataloaders
    # -----------------------------
    train_loader = DataLoader(LungCancerDataset(dfs['train'], train_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(LungCancerDataset(dfs['val'], val_test_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(LungCancerDataset(dfs['test'], val_test_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # -----------------------------
    # Model, Loss, Optimizer
    # -----------------------------
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # -----------------------------
    # Training Loop
    # -----------------------------
    best_val_acc = 0.0
    best_model_state = None
    start_time = time.time()

    for epoch in range(1, epochs+1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"Best so far: {best_val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            save_path = os.path.join(output_dir, "best_model.pth")

            checkpoint = {
                "model_state": best_model_state,
                "model_name": model_name,
                "classes": class_names
            }
            torch.save(checkpoint, save_path)
            print(f"Saved best model checkpoint to {save_path}")

    print(f"\nTraining finished in {(time.time()-start_time)/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # -----------------------------
    # Test Evaluation
    # -----------------------------
    if best_model_state:
        model.load_state_dict(best_model_state)
        test_loss, test_acc, test_f1 = validate_one_epoch(model, test_loader, criterion, device)
        print(f"Test Set | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

        y_true, y_pred = get_all_predictions(model, test_loader, device)
        target_names = [id2label[i] for i in range(num_classes)]
        report = classification_report(y_true, y_pred, target_names=target_names)
        print(report)
        with open(os.path.join(output_dir, "classification_report.txt"), 'w') as f:
            f.write(report)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.show()

        model.to("cpu").eval()
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        traced = torch.jit.trace(model, dummy_input)
        traced.save(os.path.join(output_dir, "best_model_scripted.pt"))
        print(f"Scripted model saved to {output_dir}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # Required for Windows
    main()
