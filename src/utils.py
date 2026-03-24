import torch
import logging
import os


def save_model(model, filepath):
    """
    Save the model to the specified file path.
    """
    torch.save(model.state_dict(), filepath)
    logging.info(f'Model saved to {filepath}')


def load_model(model, filepath):
    """
    Load the model from the specified file path.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()
    logging.info(f'Model loaded from {filepath}')


def get_device():
    """
    Get the current device (GPU or CPU).
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_logging(log_file='app.log'):
    """
    Setup logging configuration.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    logging.info('Logging is set up.')


# Example usage:
if __name__ == '__main__':
    setup_logging()
    device = get_device()
    logging.info(f'Using device: {device}')