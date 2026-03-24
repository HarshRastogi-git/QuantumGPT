class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, dataloader, device):
        self.model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader, device):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()