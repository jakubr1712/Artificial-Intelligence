# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Loading the breast cancer dataset
breast_cancer_data =  pd.read_csv('./breast_cancer_data.csv')
# Printing the first sample of the dataset
breast_cancer_data

# Wyodrębnienie kolumny 'target'
labels = breast_cancer_data['target']

# Usunięcie kolumny 'target' z głównego zbioru danych
features = breast_cancer_data.drop('target', axis=1)

# Sprawdzenie pierwszych kilku wierszy danych
print(breast_cancer_data.head())

# Sprawdzenie brakujących wartości
print(breast_cancer_data.isnull().sum())

# Podstawowe statystyki danych
print(breast_cancer_data.describe())

# Sprawdzenie balansu klas w kolumnie 'target'
print(breast_cancer_data['target'].value_counts())


# Inicjalizacja skalera
scaler = StandardScaler()

# Dopasowanie skalera do cech i ich transformacja
scaled_features = scaler.fit_transform(features)

# Zamiana z powrotem na DataFrame
scaled_features_df = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

# Sprawdzenie statystyk po normalizacji
print("po normalizacji")
print(scaled_features_df.describe())

# Wizualizacja rozkładu wartości dla kilku cech
breast_cancer_data[['mean radius', 'mean texture', 'mean perimeter', 'mean area']].hist(bins=20)
# Wizualizacja rozkładu wartości po normalizacji
scaled_features_df[['mean radius', 'mean texture', 'mean perimeter', 'mean area']].hist(bins=20)
plt.show()


# Splitting the dataset into a training set and a validation set
training_data, validation_data, training_labels, validation_labels = train_test_split(scaled_features_df, labels, test_size = 0.2, random_state = 77)

# Printing the shapes of the training set and the validation set
print(training_data.shape, training_labels.shape)
print(validation_data.shape, validation_labels.shape)


# Creating a KNN classifier with k=6
classifier = KNeighborsClassifier(n_neighbors = 6)

# Training the classifier with the training set
classifier.fit(training_data, training_labels)


# Evaluating the performance of the classifier on the validation set
classifier.score(validation_data, validation_labels)

# Plotting the accuracy of the classifier for different values of k
plt.figure(figsize = (10, 6))

accuracies = []

for k in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

k_list = range(1, 100)

the_best_k = np.argmax(accuracies)+1
print("Najlepsze K=",the_best_k)

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()

# Creating a KNN classifier with the best value of k found
classifier = KNeighborsClassifier(n_neighbors = the_best_k)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

# Making predictions on a sample of the validation set
print(
classifier.predict(validation_data[80:100])
)

# Printing the true labels of the sample
print(
np.array(validation_labels[80:100])
)