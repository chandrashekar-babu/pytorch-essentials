import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm, trange

class Trainer:
    def __init__(self, model=None, device=None, 
                 criterion=None, optimizer=None, 
                 train_data_path=None, transform=None,
                 load_saved_model=False, model_save_path=None):

        if device:
            self._device = device
        else:
            self._device = torch.accelerator.current_accelerator() \
                if torch.accelerator.is_available() else torch.device('cpu')

        self._train_data_path = train_data_path
        self._transform = transform
        
        self._load_saved_model = load_saved_model
        self._path = model_save_path
        
        self._model = model
        
        self._criterion = criterion
        self._optimizer = optimizer
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    # Device property
    @property
    def device(self):
        return self._device

    # Model property
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model

    # Criterion property
    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion

    # Optimizer property
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    # Transform property
    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    # Helper method to load pytorch datasets
    def load_dataset(self, dataset):
        self.train_data = dataset(root=self._train_data_path, train=True, download=True)
        self.test_data = dataset(root=self._train_data_path, train=False, download=True)

        self.X_train, self.y_train = self.transform(self.train_data)
        self.X_test, self.y_test = self.transform(self.test_data)

    def load_model(self):
        if self._path is None:
            raise ValueError("Failed to load Model: Model path is not set.")
        self.model.load_state_dict(torch.load(self._path))
        self.model.to(self.device)

    def __enter__(self):
        if self._load_saved_model:
            self.load_model()
        elif self.model is not None:
            self.model.to(self.device)
        else:
            raise ValueError("Failed to initialize Model: Model is not set.")
        return self

    def save_model(self):
        if self._path is None:
            raise ValueError("Model could not be saved: Model path is not set.")
        torch.save(self.model.state_dict(), self._path)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and self._path:
            self.save_model()

    def calculate_accuracy(self, predicted, target):
        """Returns accuracy for a batch."""
        pred = predicted.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum()
        accuracy = correct.float() / target.shape[0]
        return accuracy.item()

    # Training method
    def train_epoch(self):
        self.model.train()
        num_samples = self.X_train.shape[0]  # Number of rows indicate the number training samples
        epoch_loss = epoch_acc = 0
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for batch in tqdm(range(0, num_samples, self.batch_size), desc="Training", leave=False):

            # Select a batch based on batch size
            end = min(batch + self.batch_size, num_samples)
            batch_idx = indices[batch:end]

            x = self.X_train[batch_idx].to(self.device)
            y = self.y_train[batch_idx].to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            epoch_loss += loss.item()
            epoch_acc += self.calculate_accuracy(y_pred, y)
            self.optimizer.step()

        epoch_loss /= (num_samples // self.batch_size)
        accuracy = epoch_acc / self.batch_size
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(accuracy)

        return epoch_loss, epoch_acc


    # Define the testing function
    def test_epoch(self):
        self.model.eval()
        epoch_loss = epoch_acc = 0
        num_samples = self.X_test.shape[0]

        with torch.inference_mode():
            for start in tqdm(range(0, num_samples, self.batch_size), desc="Testing", leave=False):
                
                end = min(start + self.batch_size, num_samples)

                x = self.X_test[start:end].to(self.device)
                y = self.y_test[start:end].to(self.device)

                y_pred = self.model(x)
                epoch_loss += self.criterion(y_pred, y).item()
                epoch_acc += self.calculate_accuracy(y_pred, y)

        epoch_loss /= (num_samples // self.batch_size) * (end - start)
        accuracy = epoch_acc / self.batch_size
        self.test_losses.append(epoch_loss)
        self.test_accuracies.append(accuracy)

        return epoch_loss, epoch_acc


    def train(self, epochs=10, batch_size=64, **optimizer_kwargs):
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = self.optimizer(self.model.parameters(), **optimizer_kwargs)
        self.criterion = self.criterion()

        for epoch in trange(epochs):
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test_epoch()
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    def plot_loss_curves(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 7))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.test_losses, label="Test Loss")
        plt.title("Training And Test Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_predictions(self, predictions=None):
        import matplotlib.pyplot as plt

        # Plot training data in blue
        plt.figure(figsize=(10, 7))
        plt.scatter(self.X_train, self.y_train, c="b", s=4, label="Training data")

        # Plot test data in green
        plt.scatter(self.X_test, self.y_test, c="y", s=4, label="Test data")

        # If predictions are provided, plot them in red
        if predictions is not None:
            plt.scatter(self.X_test, predictions, c="r", s=4, label="Predictions")

        plt.legend(prop={"size": 14})
        plt.show()

    def plot_accuracy_curves(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 7))
        plt.plot(self.train_accuracies, label="Training Accuracy")
        plt.plot(self.test_accuracies, label="Test Accuracy")
        plt.title("Training And Test Accuracy Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

# Define the neural network architecture
class SimpleFFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 392)
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 49)
        self.fc4 = nn.Linear(49, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if __name__ == "__main__":
    def flatten_and_normalize(images):
        numel = images.data[0].numel()
        return images.data.view(-1, numel).float() / 255.0, images.targets


    with Trainer(model=SimpleFFNN(), 
             criterion=nn.CrossEntropyLoss, optimizer=optim.Adam, model_save_path="model.pth",
             train_data_path="Samples/Datasets", transform=flatten_and_normalize) as trainer:
        trainer.load_dataset(MNIST)
        trainer.train(epochs=10, lr=0.001)