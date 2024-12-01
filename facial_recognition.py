import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import numpy as np

# Load the dataset
print("Loading the dataset...")
lfw_dataset = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
_, h, w = lfw_dataset.images.shape  # Image dimensions
X = lfw_dataset.data  # Feature matrix
y = lfw_dataset.target  # Target labels
target_names = lfw_dataset.target_names  # Class names

print(f"Dataset loaded. Number of samples: {len(X)}")
print(f"Number of classes: {len(target_names)}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform PCA
n_components = 100  # Number of eigenfaces to keep
print(f"Performing PCA to retain the top {n_components} components...")
pca = PCA(n_components=n_components, whiten=True, random_state=42)
pca.fit(X_train)

# Transform the data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA completed. Shape of PCA-transformed data:", X_train_pca.shape)

# Train a Neural Network Classifier
print("Training the neural network classifier...")
clf = MLPClassifier(hidden_layer_sizes=(128,), batch_size=256, verbose=True, early_stopping=True, random_state=42)
clf.fit(X_train_pca, y_train)

# Test the Classifier
print("Evaluating the classifier...")
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))

# Function to plot images
def plot_gallery(images, titles, h, w, rows=3, cols=4):
    plt.figure(figsize=(10, 6))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=10)
        plt.xticks(())
        plt.yticks(())

# Titles for predictions
def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield f"pred: {pred_name}\ntrue: {true_name}"

# Visualize predictions
prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, h, w)

# Visualize Eigenfaces
eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = [f"Eigenface {i}" for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
