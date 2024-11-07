import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ramanspy as rp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "./data"

print("Loading data...")
# Load training and testing datasets
X_train, y_train = rp.datasets.bacteria("train", data_path)
X_test, y_test = rp.datasets.bacteria("test", data_path)

# Load the names of the species and antibiotics corresponding to the 30 classes
y_labels, antibiotics_labels = rp.datasets.bacteria("labels")

X_train_tensor = torch.tensor(X_train.spectral_data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.spectral_data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print("Data loaded!")

# Create DataLoader for training and testing datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define the CNN with Residual Connections
class ResidualCNN(nn.Module):
    def __init__(self, num_classes):
        super(ResidualCNN, self).__init__()
        self.layer1 = self._make_layer(1, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.layer5 = self._make_layer(512, 512, stride=2)
        self.layer6 = self._make_layer(512, 512, stride=2)
        self.layer7 = self._make_layer(512, 512, stride=2)
        self.layer8 = self._make_layer(512, 512, stride=2)
        self.layer9 = self._make_layer(512, 512, stride=2)
        self.layer10 = self._make_layer(512, 512, stride=2)
        self.layer11 = self._make_layer(512, 512, stride=2)
        self.layer12 = self._make_layer(512, 512, stride=2)
        self.layer13 = self._make_layer(512, 512, stride=2)
        self.layer14 = self._make_layer(512, 512, stride=2)
        self.layer15 = self._make_layer(512, 512, stride=2)
        self.layer16 = self._make_layer(512, 512, stride=2)
        self.layer17 = self._make_layer(512, 512, stride=2)
        self.layer18 = self._make_layer(512, 512, stride=2)
        self.layer19 = self._make_layer(512, 512, stride=2)
        self.layer20 = self._make_layer(512, 512, stride=2)
        self.layer21 = self._make_layer(512, 512, stride=2)
        self.layer22 = self._make_layer(512, 512, stride=2)
        self.layer23 = self._make_layer(512, 512, stride=2)
        self.layer24 = self._make_layer(512, 512, stride=2)
        self.layer25 = self._make_layer(512, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        return ResidualBlock(in_channels, out_channels, stride)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        x = self.layer25(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Hyperparameters
input_dim = X_train_tensor.shape[1]
output_dim = len(y_labels)  # Assuming y_labels contains the unique classes

# Instantiate the model, loss function, and optimizer
model = ResidualCNN(output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")

# Training loop
num_epochs = 50
scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    i = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        print(f"PROCESSED {i}")
        i += 1
    
    # Calculate validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            with torch.cuda.amp.autocast():  # Mixed precision training
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(test_loader)}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

# Save the model
torch.save(model.state_dict(), "residual_cnn_model.pth")