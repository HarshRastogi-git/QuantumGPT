class TextDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        # Load your dataset here

    # Implement your dataset methods here


def create_dataloader(dataset, batch_size=32, shuffle=True):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)