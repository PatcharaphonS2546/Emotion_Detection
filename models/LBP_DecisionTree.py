import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.metrics import accuracy_score

#แก้ต่อจากตรงนี้

class LBP_DecisionTree:
    def __init__(self, radius=1, n_points=8):
        self.radius = radius
        self.n_points = n_points
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        # ตรวจสอบว่าภาพมี 3D หรือ 4D แล้วแปลงเป็น 2D
        X_train_lbp = [
            local_binary_pattern(img.squeeze(), self.n_points, self.radius, method="uniform").ravel()
            for img in X_train
        ]

        X_train_lbp = np.array(X_train_lbp)

        start_time = time.time()

        self.model.fit(X_train_lbp, y_train)
        self.train_time = time.time() - start_time

    def evaluate(self, X_test, y_test):
        X_test_lbp = [local_binary_pattern(img.squeeze(), self.n_points, self.radius, method="uniform").ravel() for img in X_test]
        y_pred = self.model.predict(X_test_lbp)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, self.train_time
