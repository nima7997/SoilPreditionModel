import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# ==============================================
# CONFIGURATION
# ==============================================
MODEL_DIR = "./models_phi_C_bootstrap"
LABELED_DATA_PATH = "NewDF - Sheet1.csv"  # replace with your real labeled dataset path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR_BASE = 1e-3          # learning rate for heads
LR_FEATURES = 1e-5      # learning rate for frozen layers (if unfreezing later)
WEIGHT_DECAY = 1e-4     # L2 regularization
EPOCHS = 300
BATCH_SIZE = 16
PATIENCE = 20            # early stopping patience
FREEZE_EPOCHS = 50       # unfreeze after this many epochs

# ==============================================
# DATA PREPARATION
# ==============================================
data = np.loadtxt(LABELED_DATA_PATH, delimiter=",", skiprows=1)
X = data[:, :-2]
y_C = data[:, -2]
y_phi = data[:, -1]
y = np.stack([y_C, y_phi], axis=1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

n_samples = len(X)
n_train = int(0.8 * n_samples)
n_hold = n_samples - n_train

train_ds, hold_ds = random_split(TensorDataset(X, y), [n_train, n_hold])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
hold_dl = DataLoader(hold_ds, batch_size=BATCH_SIZE)

print(f"Labeled rows: {n_samples}. Train: {n_train}, Holdout: {n_hold}")

# ==============================================
# MODEL DEFINITION
# ==============================================
class FewShotModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # Shared feature extractor (pretrained backbone)
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Separate heads for C and phi
        self.C_head = nn.Linear(hidden_dim, 1)
        self.phi_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feats = self.features(x)
        C_pred = self.C_head(feats)
        phi_pred = self.phi_head(feats)
        return torch.cat([C_pred, phi_pred], dim=1)

# ==============================================
# LOAD PRETRAINED WEIGHTS (optional)
# ==============================================
# if you have a pretrained model saved:
pretrained_path = os.path.join(MODEL_DIR, "fewshot_pretrained.pt")
model = FewShotModel(input_dim=X.shape[1]).to(DEVICE)
if os.path.exists(pretrained_path):
    print(f"Loading pretrained model from {pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))

# ==============================================
# TRAINING LOOP
# ==============================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY)
best_loss = float("inf")
patience_counter = 0

# Freeze feature extractor initially
for param in model.features.parameters():
    param.requires_grad = False

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_dl.dataset)

    # Holdout evaluation
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in hold_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            truths.append(yb.cpu().numpy())

    y_pred_hold = np.concatenate(preds)
    y_hold = np.concatenate(truths)
    hold_loss = mean_squared_error(y_hold, y_pred_hold)

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Holdout MSE: {hold_loss:.4f}")

    # Unfreeze gradually
    if epoch == FREEZE_EPOCHS:
        print("Unfreezing feature extractor for fine-tuning...")
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.Adam([
            {"params": model.features.parameters(), "lr": LR_FEATURES},
            {"params": model.C_head.parameters()},
            {"params": model.phi_head.parameters()},
        ], lr=LR_BASE, weight_decay=WEIGHT_DECAY)

    # Early stopping
    if hold_loss < best_loss:
        best_loss = hold_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fewshot_best.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ==============================================
# FINAL EVALUATION
# ==============================================
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "fewshot_best.pt")))
model.eval()
y_pred_hold = []
y_hold = []
with torch.no_grad():
    for xb, yb in hold_dl:
        pred = model(xb.to(DEVICE))
        y_pred_hold.append(pred.cpu().numpy())
        y_hold.append(yb.numpy())
y_pred_hold = np.concatenate(y_pred_hold)
y_hold = np.concatenate(y_hold)

for i, name in enumerate(["C", "phi"]):
    mae = mean_absolute_error(y_hold[:, i], y_pred_hold[:, i])
    rmse = mean_squared_error(y_hold[:, i], y_pred_hold[:, i])
    r2 = r2_score(y_hold[:, i], y_pred_hold[:, i])
    print(f"Holdout {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

print("✅ Training complete. Best model saved to fewshot_best.pt")
