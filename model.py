import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load data from JSON
with open('data.json', 'r') as f:
    raw_data = json.load(f)

# Function to convert a single data point
def convert_data_point(data_point):
    wl = np.array(data_point["wl"])
    r = np.array(data_point["r"])
    c = np.array([item[2] for item in data_point["l"]])  # The third element in each 'l' entry is the concentration

    return wl, r, c

# Prepare data lists
X = []
y = []
wl = None

for data_point in raw_data:
    wl_, r, c = convert_data_point(data_point)
    X.append(r)  # r vector (1000 values wide)
    y.append(c)  # c vector (12 values wide)
    wl = wl_     # wl is not used in the neural network but could be useful for other purposes

X = preprocessing.normalize(np.array(X))
y = preprocessing.normalize(np.array(y))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network model
class NNet(nn.Module):
    def __init__(self, input_size=100, output_size=12):
        super(NNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# Initialize model, loss, and optimizer
model = NNet(100, 12)
criterion = rmse  # nn.MSELoss() is another option
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initial setup
best_loss = math.inf
patience = 100
counter = 0
epoch = 1

# Training loop
while counter < patience:
    model.train()
    # Forward pass on training set
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Check for improvement
    if loss < best_loss:
        best_loss = loss
        counter = 0  # Reset counter if there's improvement
    else:
        counter += 1  # Increment counter if no improvement
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}], Loss: {best_loss:.10f}")
    epoch += 1

    if epoch > 100:
        break

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss:.10f}")

# Testing with the first sample from the test set
with torch.no_grad():
    test_output = model(X_test[0].unsqueeze(0))  # Add batch dimension for a single sample
    print("Predicted output for the first test sample:", test_output)
    print("Actual output for the first test sample:", y_test[0])
