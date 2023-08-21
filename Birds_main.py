from functions import train_model
import torchvision.transforms as transforms

def main():
    # Example usage:

    # 1. Train the model
    train_path = "/path/to/train_data"
    valid_path = "/path/to/valid_data"
    model_save_path = "/path/to/save/model.pth"
    trained_model, train_accuracies, val_accuracies = train_model(train_path, valid_path, num_epochs=5, model_save_path=model_save_path)

    # Print training and validation accuracies for reference
    print("Training Accuracies:", train_accuracies)
    print("Validation Accuracies:", val_accuracies)

if __name__ == "__main__":
    main()
