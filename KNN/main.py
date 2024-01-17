# Importowanie potrzebnych bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Wczytywanie zbioru danych dotyczących raka piersi
breast_cancer_data = pd.read_csv('breast_cancer_data.csv')

# Wyświetlenie pierwszych wierszy zbioru danych
print(breast_cancer_data.head())

# Wyodrębnienie etykiet i cech
labels = breast_cancer_data['target']
features = breast_cancer_data.drop('target', axis=1)

# Sprawdzenie balansu klas w kolumnie 'target'
malignant = (labels == 0).sum()
benign = (labels == 1).sum()
print(f"Liczba przypadków złośliwych: {malignant}")
print(f"Liczba przypadków łagodnych: {benign}")

# Sprawdzenie, czy są jakieś brakujące wartości
print(breast_cancer_data.isnull().sum())

# Wyświetlenie podstawowych statystyk zbioru danych
print(breast_cancer_data.describe())

# Inicjalizacja skalera
scaler = StandardScaler()

# Dopasowanie skalera do cech i ich transformacja
scaled_features = scaler.fit_transform(features)

# Zamiana przeskalowanych cech z powrotem na DataFrame
scaled_features_df = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

# Wyświetlenie statystyk po normalizacji
print("Statystyki po normalizacji:")
print(scaled_features_df.describe())

# Wizualizacja rozkładu wartości cech przed normalizacją
breast_cancer_data[['mean radius', 'mean texture', 'mean perimeter', 'mean area']].hist(bins=20)
plt.suptitle("Rozkład wartości cech przed normalizacją")

# Wizualizacja rozkładu wartości cech po normalizacji
scaled_features_df[['mean radius', 'mean texture', 'mean perimeter', 'mean area']].hist(bins=20)
plt.suptitle("Rozkład wartości cech po normalizacji")
plt.show()

# Podział danych na zestawy treningowe i walidacyjne
training_data, validation_data, training_labels, validation_labels = train_test_split(
    scaled_features_df, labels, test_size=0.2, random_state=77)

# Wydrukowanie rozmiarów zestawów treningowych i walidacyjnych
print("Rozmiar danych treningowych:", training_data.shape)
print("Rozmiar danych walidacyjnych:", validation_data.shape)

# Tworzenie klasyfikatora KNN z k=6
classifier = KNeighborsClassifier(n_neighbors=6)

# Trenowanie klasyfikatora na danych treningowych
classifier.fit(training_data, training_labels)

# Ocena wydajności klasyfikatora na zestawie walidacyjnym
accuracy = classifier.score(validation_data, validation_labels)
print(f'Dokładność klasyfikatora na zestawie walidacyjnym: {accuracy:.2f}')

# Wykres dokładności klasyfikatora dla różnych wartości k
plt.figure(figsize=(10, 6))
accuracies = []
for k in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

k_list = range(1, 100)
the_best_k = np.argmax(accuracies) + 1
print("Najlepsze k:", the_best_k)

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Dokładność walidacji")
plt.title("Dokładność klasyfikatora raka piersi")
plt.show()

# Tworzenie klasyfikatora KNN z najlepszą znalezioną wartością k
classifier = KNeighborsClassifier(n_neighbors=the_best_k)
classifier.fit(training_data, training_labels)
print("Dokładność klasyfikatora z najlepszym k:", classifier.score(validation_data, validation_labels))

# Wykonanie predykcji na próbce danych walidacyjnych
predictions = classifier.predict(validation_data[80:100])
print("Predykcje klasyfikatora:", predictions)

# Wyświetlenie prawdziwych etykiet próbki
true_labels = np.array(validation_labels[80:100])
print("Prawdziwe etykiety:     ", true_labels)

input("Naciśnij Enter, aby zakończyć program.")
