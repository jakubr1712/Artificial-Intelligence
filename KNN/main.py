# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load breast cancer dataset
breast_cancer_data = pd.read_csv('breast_cancer_data.csv')

# Display the first rows of the dataset
print(breast_cancer_data.head())

# Extract labels and features
labels = breast_cancer_data['target']
features = breast_cancer_data.drop('target', axis=1)

# Check class balance in the 'target' column
malignant = (labels == 0).sum()
benign = (labels == 1).sum()
print(f"Number of malignant cases: {malignant}")
print(f"Number of benign cases: {benign}")

# Check for any missing values
print(breast_cancer_data.isnull().sum())

# Display basic dataset statistics
print(breast_cancer_data.describe())

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to features and transform them
scaled_features = scaler.fit_transform(features)

# Convert scaled features back to a DataFrame
scaled_features_df = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

# Display statistics after normalization
print("Statistics after normalization:")
print(scaled_features_df.describe())

# Visualize the distribution of feature values before normalization
breast_cancer_data[['mean radius', 'mean texture', 'mean perimeter', 'mean area']].hist(bins=20)
plt.suptitle("Distribution of feature values before normalization")

# Visualize the distribution of feature values after normalization
scaled_features_df[['mean radius', 'mean texture', 'mean perimeter', 'mean area']].hist(bins=20)
plt.suptitle("Distribution of feature values after normalization")
plt.show()

# Split the data into training and validation sets
training_data, validation_data, training_labels, validation_labels = train_test_split(
    scaled_features_df, labels, test_size=0.2, random_state=77)

# Print the sizes of training and validation sets
print("Training data size:", training_data.shape)
print("Validation data size:", validation_data.shape)

# Create a KNN classifier with k=6
classifier = KNeighborsClassifier(n_neighbors=6)

# Train the classifier on the training data
classifier.fit(training_data, training_labels)

# Evaluate the classifier's performance on the validation set
accuracy = classifier.score(validation_data, validation_labels)
print(f'Classifier accuracy on the validation set: {accuracy:.2f}')

# Plot the accuracy of the classifier for different values of k
plt.figure(figsize=(10, 6))
accuracies = []
for k in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

k_list = range(1, 100)
the_best_k = np.argmax(accuracies) + 1
print("Best k:", the_best_k)

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()

# Create a KNN classifier with the best-found k value
classifier = KNeighborsClassifier(n_neighbors=the_best_k)
classifier.fit(training_data, training_labels)
print("Classifier accuracy with the best k:", classifier.score(validation_data, validation_labels))

# Make predictions on a subset of validation data
predictions = classifier.predict(validation_data[80:100])
print("Classifier predictions:", predictions)

# Display the true labels of the subset
true_labels = np.array(validation_labels[80:100])
print("True labels:     ", true_labels)

