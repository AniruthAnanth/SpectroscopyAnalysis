import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
with open('data.json', 'r') as f:
    raw_data = json.load(f)

def convert_data_point(data_point):
    wl = np.array(data_point["wl"])
    r = np.array(data_point["r"])
    c = np.array([item[2] for item in data_point["l"]])
    return wl, r, c

X, y, wl = [], [], None
for data_point in raw_data:
    wl_, r, c = convert_data_point(data_point)
    X.append(r)
    y.append(c)
    wl = wl_

X = preprocessing.normalize(np.array(X))
y = preprocessing.normalize(np.array(y))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

class Encoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, y, X):
        x = torch.cat([y, X], dim=1)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, X):
        x = torch.cat([z, X], dim=1)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        y_recon = torch.sigmoid(self.fc_out(h))
        return y_recon

class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, cond_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, X):
        mu, logvar = self.encoder(y, X)
        z = self.reparameterize(mu, logvar)
        y_recon = self.decoder(z, X)
        return y_recon, mu, logvar

# Loss function for CVAE
def loss_function(recon_y, y, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_y, y, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss * 0.5

# Hyperparameters
input_dim = y_train.shape[1]
cond_dim = X_train.shape[1]
hidden_dim = 256
latent_dim = 8

# Instantiate model and optimizer
model = CVAE(input_dim, cond_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Early stopping parameters
patience = 15
best_val_loss = float('inf')
trigger_times = 0

# Training with early stopping
num_epochs = 1000
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y_recon, mu, logvar = model(y_batch, X_batch)
            loss = loss_function(y_recon, y_batch, mu, logvar)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            y_recon, mu, logvar = model(y_val_batch, X_val_batch)
            val_loss += loss_function(y_recon, y_val_batch, mu, logvar).item()

    val_loss /= len(val_loader.dataset)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader.dataset):.8f}, Val Loss: {val_loss:.8f}')

    # Early stopping check
    if val_loss < best_val_loss or epoch < 150:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered")
            break

# Function to estimate y given just X using the trained CVAE's decoder
def estimate_y(X):
    model.eval()
    with torch.no_grad():
        z = torch.randn((X.size(0), latent_dim)).to(device)
        y_estimated = model.decoder(z, X)
    return y_estimated

def mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

torch.save(model.state_dict(), "v5_weights.pth")

# Testing the estimation function
y_pred = estimate_y(X_val)
print("Estimated y values:", mae(y_pred, y_val).item())