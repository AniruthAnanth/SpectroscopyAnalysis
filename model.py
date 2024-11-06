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

with open('data.json', 'r') as f:
    raw_data = json.load(f)

def convert_data_point(data_point):
    wl = np.array(data_point["wl"])
    r = np.array(data_point["r"])
    c = np.array([item[2] for item in data_point["l"]])
    return wl, r, c

def calculate_deviation(y_real, y_pred):
    # Calculate absolute errors and mean absolute deviation (MAD)
    abs_errors = torch.abs(y_pred - y_real)
    mean_abs_deviation = torch.mean(abs_errors)
    
    # Calculate standard deviation of errors
    errors = y_pred - y_real
    mean_error = torch.mean(errors)
    std_dev_error = torch.sqrt(torch.mean((errors - mean_error) ** 2))

    return mean_abs_deviation, std_dev_error

X = []
y = []
wl = None

for data_point in raw_data:
    wl_, r, c = convert_data_point(data_point)
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

class OptimizedNNet(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(OptimizedNNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=1)
        )
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )
        self.output_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        attention = self.attention_layer(x) * x
        x = attention + self.hidden_layers(x)
        x = self.output_layer(x)
        return x

def mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

model = OptimizedNNet(100, 256, 12)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
batch_size = 64

best_loss = math.inf
epoch = 0
grad_clip_value = 1.0

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

try:
    while True:
        start = time.time()
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            if loss < best_loss:
                best_loss = loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()
        with torch.no_grad():
            #if (epoch + 1) % 1 == 0 or epoch == 0:
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                mad, std_dev = calculate_deviation(y_test, test_outputs)
                print(f"Epoch [{epoch + 1}], Loss: {loss:.10f}, Test: {test_loss:.6f}, Best: {best_loss:.6f}, LR: {float(scheduler.get_last_lr()[0]):.2f}, MAD: ±{mad * 100:.6f}% in {time.time() - start:.4f}s")
                scheduler.step(test_loss)
        epoch += 1
except KeyboardInterrupt:
    print()
    print("Broken out of loop!")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss:.10f}")



with torch.no_grad():
    test_outputs = model(X_test)

    mad, std_dev = calculate_deviation(y_test, test_outputs)
    print(f"Mean Absolute Deviation (MAD): ±{mad * 100:.3f}%")
    #print(f"Standard Deviation of Errors: ±{std_dev:.4f}")
    #print("Predicted output for the first test sample:", test_output)
    #print("Actual output for the first test sample:", y_test[0])
