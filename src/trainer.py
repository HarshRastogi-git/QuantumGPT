import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, criterion, optimizer, device='cpu'):
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0

            # Training loop
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            # Log training loss
            avg_loss = running_loss / len(self.train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            self.validate()  # Validate after each epoch

    def validate(self):
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                # Compute accuracy
                predicted = outputs.argmax(dim=1)
                correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

# Example usage:
# model = ...  # Your model definition
# train_dataset = ...  # Your training dataset
# val_dataset = ...  # Your validation dataset
# criterion = nn.CrossEntropyLoss()  # Loss function
# optimizer = torch.optim.Adam(model.parameters())  # Optimizer
# trainer = Trainer(model, train_dataset, val_dataset, criterion, optimizer)
# trainer.train(epochs=10)