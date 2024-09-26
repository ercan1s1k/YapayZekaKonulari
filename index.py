# Gerekli kütüphaneleri yükleyin
import numpy as np  # Sayısal işlemler için
import pandas as pd  # Veri işleme için
import matplotlib.pyplot as plt  # Veri görselleştirme için
import seaborn as sns  # Gelişmiş veri görselleştirme için
from sklearn.model_selection import train_test_split  # Veri setini bölmek için
from sklearn.preprocessing import StandardScaler  # Verileri ölçeklendirmek için
from sklearn.linear_model import LogisticRegression  # Basit bir makine öğrenimi modeli
from sklearn.metrics import accuracy_score, classification_report  # Modelin doğruluğunu değerlendirmek için
from sklearn.decomposition import PCA  # Boyut indirgeme için
from sklearn.cluster import KMeans  # Kümeleme algoritması
from sklearn.ensemble import RandomForestClassifier  # Gelişmiş makine öğrenimi modeli
import tensorflow as tf  # Derin öğrenme için
from tensorflow.keras.models import Sequential  # Derin öğrenme modeli oluşturmak için
from tensorflow.keras.layers import Dense, Dropout  # Derin öğrenme katmanları
from tensorflow.keras.preprocessing.text import Tokenizer  # NLP için metin işleme
from tensorflow.keras.preprocessing.sequence import pad_sequences  # NLP için dizileri pad etmek
from tensorflow.keras.datasets import mnist  # Görüntü işleme için veri seti
from tensorflow.keras.utils import to_categorical  # Kategorik verileri işlemek için
import gym  # Pekiştirmeli öğrenme için

# Veri setini yükleyin (örnek olarak iris veri seti kullanılıyor)
from sklearn.datasets import load_iris
iris = load_iris()

# Verileri ve hedef değişkeni ayırın
X = iris.data  # Özellikler
y = iris.target  # Hedef değişken

# Veriyi eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri ölçeklendirin
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lojistik regresyon modelini oluşturun ve eğitin
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred = log_reg_model.predict(X_test)

# Modelin doğruluğunu hesaplayın
accuracy = accuracy_score(y_test, y_pred)
print(f"Lojistik Regresyon Modeli Doğruluğu: {accuracy * 100:.2f}%")

# Boyut indirgeme (PCA) uygulayın
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Kümeleme (KMeans) uygulayın
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_pca)
clusters = kmeans.predict(X_pca)

# Kümeleme sonuçlarını görselleştirin
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('KMeans Kümeleme Sonuçları')
plt.show()

# Gelişmiş makine öğrenimi modeli (Random Forest) oluşturun ve eğitin
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred_rf = rf_model.predict(X_test)

# Modelin doğruluğunu hesaplayın
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Modeli Doğruluğu: {accuracy_rf * 100:.2f}%")

# Derin öğrenme modeli oluşturun ve eğitin (MNIST veri seti ile)
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
X_train_mnist = X_train_mnist.reshape(-1, 28*28) / 255.0
X_test_mnist = X_test_mnist.reshape(-1, 28*28) / 255.0
y_train_mnist = to_categorical(y_train_mnist)
y_test_mnist = to_categorical(y_test_mnist)

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_mnist, y_train_mnist, epochs=10, validation_split=0.2)

# Modelin doğruluğunu hesaplayın
loss, accuracy_dl = model.evaluate(X_test_mnist, y_test_mnist)
print(f"Derin Öğrenme Modeli Doğruluğu: {accuracy_dl * 100:.2f}%")

# Doğal Dil İşleme (NLP) örneği
texts = ["Merhaba dünya", "Yapay zeka harika", "Python ile NLP"]
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=5)

print("Tokenize edilmiş ve pad edilmiş diziler:")
print(padded_sequences)

# Pekiştirmeli öğrenme örneği (CartPole-v1 ortamı)
env = gym.make('CartPole-v1')
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Rastgele bir eylem seç
    next_state, reward, done, info = env.step(action)
    state = next_state
    env.render()

env.close()
