import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns


class PCA_KNN:
    def __init__(self, n_components=100, n_neighbors=5):
        self.pca = PCA(n_components=n_components)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()  # ใช้แปลง labels เป็นตัวเลข
        self.train_time = None

    def train(self, X_train, y_train):
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_train = self.scaler.fit_transform(X_train)
        X_train_pca = self.pca.fit_transform(X_train)

        y_train = self.encoder.fit_transform(y_train)  # แปลง labels เป็นตัวเลข

        start_time = time.time()
        self.knn.fit(X_train_pca, y_train)
        self.train_time = time.time() - start_time

        # Plot PCA ถ้า n_components >= 2
        if self.pca.n_components_ >= 2:
            self.plot_pca(X_train_pca, y_train)

    def plot_pca(self, X_pca, y):
        plt.figure(figsize=(12, 6))

        # 2D Scatter plot
        if X_pca.shape[1] >= 2:
            plt.subplot(1, 2, 1)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
            plt.title("PCA - First 2 Components")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")

        # 3D Scatter plot (ถ้ามีอย่างน้อย 3 มิติ)
        if X_pca.shape[1] >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.subplot(1, 2, 2, projection='3d')
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.6)
            ax.set_title("PCA - First 3 Components")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.set_zlabel("PC 3")

        plt.tight_layout()
        plt.show()

    def evaluate(self, X_test, y_test):
        X_test = X_test.reshape(X_test.shape[0], -1)
        X_test = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test)

        y_test = self.encoder.transform(y_test)  # แปลง labels เป็นตัวเลข

        y_pred = self.knn.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.encoder.classes_,
                    yticklabels=self.encoder.classes_)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

        return accuracy, self.train_time

    def predict(self, X_input):
        X_input = X_input.reshape(1, -1)
        X_input = self.scaler.transform(X_input)
        X_input_pca = self.pca.transform(X_input)
        label_index = self.knn.predict(X_input_pca)
        return self.encoder.inverse_transform(label_index)  # แปลงกลับเป็นชื่อคลาส

    def save_model(self, filepath="pca_knn_model.pkl"):
        with open(filepath, "wb") as f:
            joblib.dump((self.pca, self.knn, self.scaler, self.encoder), f)

    @classmethod
    def load_model(cls, filepath="pca_knn_model.pkl"):
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            raise FileNotFoundError(f"Model file '{filepath}' not found or is empty. Train and save the model first.")

        with open(filepath, "rb") as f:
            try:
                components = joblib.load(f)
                if len(components) != 4:
                    raise ValueError("Model file does not contain the correct number of components.")
                pca, knn, scaler, encoder = components
            except ValueError as e:
                raise ValueError(f"Error loading model: {str(e)}")

        # If successful, instantiate the model and assign the components
        model = cls(n_components=pca.n_components, n_neighbors=knn.n_neighbors)
        model.pca = pca
        model.knn = knn
        model.scaler = scaler
        model.encoder = encoder
        return model
