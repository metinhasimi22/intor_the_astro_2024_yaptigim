import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def deep_learning_classifier(x, y, epochs=50, batch_size=32):
    # Veriyi eğitim ve test setlerine ayır
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Eğer hedef değişken kategorikse, one-hot encoding yap
    if len(np.unique(y)) > 2:  # Çok sınıflı durumlar
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    # Modeli oluştur
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax' if len(np.unique(y)) > 2 else 'sigmoid'))

    # Modeli derle
    model.compile(optimizer='adam', loss='categorical_crossentropy' if len(np.unique(y)) > 2 else 'binary_crossentropy', metrics=['accuracy'])

    # Modeli eğit
    print("Model eğitiliyor...")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Tahmin yap
    predictions = model.predict(x_test)
    
    # Tahminleri sınıflara dönüştür
    predicted_classes = np.argmax(predictions, axis=1) if len(np.unique(y)) > 2 else (predictions > 0.5).astype(int)

    # Performans ölçümleri
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Doğruluk: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test.argmax(axis=1) if len(np.unique(y)) > 2 else y_test, predicted_classes))
    print("Classification Report:")
    print(classification_report(y_test.argmax(axis=1) if len(np.unique(y)) > 2 else y_test, predicted_classes))

# Örnek kullanım
# x ve y veri setlerinizi tanımlayın ve aşağıdaki fonksiyonu çağırın
# deep_learning_classifier(x, y, epochs=50, batch_size=32)