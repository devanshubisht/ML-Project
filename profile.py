# Import packages
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class Model:  
    """
    This class represents an AI model.
    """
    
    def __init__(self):
        """
        Constructor for Model class.
  
        Parameters
        ----------
        self : object
            The instance of the object passed by Python.
        """
        # TODO: Replace the following code with your own initialization code.
        self.model = None

    def nearest_neighbor_interpolation(self, image):
        """
        Perform nearest neighbor interpolation for a single image.
        image: A single image with shape (C, H, W).
        """
        C, H, W = image.shape
        for c in range(C):
            channel = image[c, :, :]
            nan_pixels = np.argwhere(np.isnan(channel))
            for y, x in nan_pixels:
                self.fill_nan_pixel(channel, y, x, H, W)
        return image

    def get_neighbors(self, y, x, H, W):
        """
        Get coordinates of neighboring pixels using a traditional for loop.
        """
        neighbors = []
        for y2 in range(y-1, y+2):
            for x2 in range(x-1, x+2):
                if 0 <= y2 < H and 0 <= x2 < W and (y2, x2) != (y, x):
                    neighbors.append((y2, x2))
        return neighbors

    def fill_nan_pixel(self, channel, y, x, H, W):
        """
        Fill NaN pixel with the mean of its neighbors.
        """
        neighbors = self.get_neighbors(y, x, H, W)
        neighbor_vals = [channel[y2, x2] for y2, x2 in neighbors if not np.isnan(channel[y2, x2])]
        if neighbor_vals:
            channel[y, x] = np.mean(neighbor_vals)

    def oversample(self, images, labels):
        minority_labels = [1, 2]
        # Find the number of instances of label 0
        num_instances_label_0 = np.sum(labels == 0)
        for minority_label in minority_labels:
            # Find indices of samples belonging to the minority class
            minority_indices = np.where(labels == minority_label)[0]
            # Calculate the number of instances to add for augmentation
            num_instances_to_add = num_instances_label_0 - len(minority_indices)
            # Randomly sample from the minority class indices to match the count of label 0
            sampled_minority_indices = np.random.choice(minority_indices, size=num_instances_to_add, replace=True)
            # Apply your data augmentation to the samples at sampled_minority_indices
            for idx in sampled_minority_indices:
                # Assuming data is in the shape (C, H, W)
                image_data = torch.tensor(images[idx])  # Convert to PyTorch tensor
                # Apply transformations (you can customize these)
                if np.random.rand() < 0.5:
                    # Random horizontal flip
                    image_data = torch.flip(image_data, dims=[2])
                if np.random.rand() < 0.5:
                    # Random rotation (clockwise)
                    angle = np.random.uniform(-30, 30)
                    image_data = torch.transpose(torch.flip(image_data, dims=[1]), 1, 2)  # Rotate 90 degrees clockwis

                # Append the augmented data to the original dataset
                images = np.concatenate([images, [image_data.numpy()]], axis=0)
                labels = np.concatenate([labels, [minority_label]])
        
        return images, labels
    
    def preprocess(self, X, y= None):
        if y is not None:
            # Remove NaN labels
            nan_mask = np.isnan(y)
            labels = y[~nan_mask]
            images = X[~nan_mask]

            images[images < 0] = 0  # Set negative pixel values to 0
            images[images > 255] = 255  # Set pixel values > 1 to 1

            print(images.shape)

            # Interpolate missing values
            images = np.array([self.nearest_neighbor_interpolation(img) for img in images])

            # Oversample to handle imbalance through augmentation
            images, labels = self.oversample(images, labels)

            return (images, labels)
        
        else:
            print("Preprocessing test data")
            X[X < 0] = 0  # Set negative pixel values to 0
            X[X > 255] = 255
            # Interpolate missing values
            images = np.array([self.nearest_neighbor_interpolation(img) for img in X])
            return (images)

    
    def fit(self, X, y):
        images, labels = self.preprocess(X, y)

        images_train = torch.tensor(images, dtype=torch.float32)
        y_train_tensor = torch.tensor(labels, dtype=torch.long)
        train_dataset = TensorDataset(images_train, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, worker_init_fn=np.random.seed)

        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
                self.act1 = nn.ReLU()
                self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
                self.act2 = nn.ReLU()
                self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
                
                self.fc = nn.Linear(64 * 4 * 4, num_classes)

            def forward(self, x):
                x = self.pool1(self.act1(self.conv1(x)))
                x = self.pool2(self.act2(self.conv2(x)))
                x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
                x = self.fc(x)
                return x


        # Instantiate the model
        num_classes = 3  # Change this based on your dataset
        self.model = SimpleCNN(num_classes)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training the model
        print("start")
        num_epochs = 50
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        return self
    
    def predict(self, X):
        """
        Use the trained model to make predictions.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, channel, height, width)
            Input data.
            
        Returns
        -------
        ndarray of shape (n_samples,)
        Predicted target values per element in X.
           
        """
        # Preprocess the test data
        X = self.preprocess(X)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Make predictions using the trained neural network
        with torch.no_grad():
            outputs = self.model(X_tensor)

        _, predicted_labels = torch.max(outputs, 1)

        # Convert PyTorch tensor to numpy array
        return predicted_labels.numpy()
    

# Load data
with open('data.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True).item()
    X = data['image']
    y = data['label']

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Filter test data that contains no labels
# In Coursemology, the test data is guaranteed to have labels
nan_indices = np.argwhere(np.isnan(y_test)).squeeze()
mask = np.ones(y_test.shape, bool)
mask[nan_indices] = False
X_test = X_test[mask]
y_test = y_test[mask]

# Train and predict
model = Model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model predition
# Learn more: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
print("F1 Score (macro): {0:.2f}".format(f1_score(y_test, y_pred, average='macro'))) # You may encounter errors, you are expected to figure out what's the issue.