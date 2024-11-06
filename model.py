import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
from scipy.interpolate import interp1d

with open('data.json', 'r') as f:
    raw_data = json.load(f)

def convert_data_point(data_point):
    wl = np.array(data_point["wl"])
    r = np.array(data_point["r"])
    c = np.array([item[2] for item in data_point["l"]])
    return wl, r, c

def calculate_deviation(y_real, y_pred):
    abs_errors = torch.abs(y_pred - y_real)
    mean_abs_deviation = torch.mean(abs_errors)
    errors = y_pred - y_real
    mean_error = torch.mean(errors)
    std_dev_error = torch.sqrt(torch.mean((errors - mean_error) ** 2))
    return mean_abs_deviation, std_dev_error

X, y, wl = [], [], None

for data_point in raw_data:
    wl_, r, c = convert_data_point(data_point)

    # Original x-axis for r (assuming r has 100 values)
    #x_original = np.linspace(0, 1, len(r))
    
    # New x-axis with 1000 points for interpolation
    #x_new = np.linspace(0, 1, 1000)
    
    # Interpolating r to 1000 values
    #interpolate_r = interp1d(x_original, r, kind='linear')
    #r_interpolated = interpolate_r(x_new)

    #X.append(r_interpolated)
    X.append(r)
    y.append(c)
    wl = wl_

X = preprocessing.normalize(np.array(X))
y = preprocessing.normalize(np.array(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class SpectralNNet(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(SpectralNNet, self).__init__()

        # 1D Convolutional Layer to capture spectral patterns (absorption spectra)
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Flatten layer to prepare for dense layers
        self.flatten = nn.Flatten()

        # Input Layer with BatchNorm and LeakyReLU
        self.input_layer = nn.Sequential(
            nn.Linear(input_size // 2 * 64, hidden_dim),  # Adjusted input size after convolution
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )

        # Attention Layer to enhance important features
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=1)
        )

        # Hidden layers with more complexity to capture non-linear relationships
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),  # Activation function focusing on self-normalization properties
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SELU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU()
        )

        # Output Layer with Sigmoid activation for chemical concentration prediction
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid()  # Ensures output stays in the [0, 1] range, assuming normalized concentrations
        )

    def forward(self, x):
        # Apply convolution to capture spectral features
        x = self.spectral_conv(x)
        x = self.flatten(x)

        x = self.input_layer(x)

        # Apply attention mechanism
        attention = self.attention_layer(x) * x

        # Skip connection with hidden layers
        x = attention + self.hidden_layers(x)

        # Final output layer
        x = self.output_layer(x)
        return x

# Custom loss functions
def mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# Instantiate and set up model
model = SpectralNNet(input_size=100, hidden_dim=128, output_size=12)
criterion = mae  # Mean Absolute Error
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
batch_size = 64
best_loss = math.inf
epoch = 0
grad_clip_value = 1.0

# Load training data
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

try:
    while True:
        start = time.time()
        model.train()
        for batch_X, batch_y in train_loader:
            # Ensure input shape is (batch_size, 1, input_size) for 1D convolution
            batch_X = batch_X.unsqueeze(1)  # Adding channel dimension for conv layer

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            if loss < best_loss:
                best_loss = loss
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()

        # Evaluate on test set
        with torch.no_grad():
            test_outputs = model(X_test.unsqueeze(1))  # Adding channel dimension
            test_loss = criterion(test_outputs, y_test)
            mad, std_dev = calculate_deviation(y_test, test_outputs)
            print(f"Epoch [{epoch + 1}], Loss: {loss:.10f}, Test: {test_loss:.6f}, Best: {best_loss:.6f}, LR: {float(scheduler.get_last_lr()[0]):.2f}, MAD: ±{mad * 100:.6f}% in {time.time() - start:.4f}s")
            scheduler.step(test_loss)
        epoch += 1
except KeyboardInterrupt:
    print("Training interrupted.")

# Final Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.unsqueeze(1))  # Adding channel dimension
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss:.10f}")

# Compute Mean Absolute Deviation (MAD) on test set
with torch.no_grad():
    test_outputs = model(X_test.unsqueeze(1))  # Adding channel dimension
    mad, std_dev = calculate_deviation(y_test, test_outputs)
    print(f"Mean Absolute Deviation (MAD): ±{mad * 100:.3f}%")

# Save model
torch.save(model.state_dict(), "v2_weights.pth")