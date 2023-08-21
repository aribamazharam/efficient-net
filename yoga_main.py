
from yoga_functions import train_yoga_model
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def main():
    # Example usage:

    # 1. Train the model
    train_path = "/path/to/train_data"
    valid_path = "/path/to/valid_data"
    model_save_path = "/path/to/save/model.pth"
    trained_model, history = train_yoga_model(train_path, valid_path, model_save_path=model_save_path)

    # Plotting the training and validation curves
    num_epochs = len(history['train_acc'])
    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
