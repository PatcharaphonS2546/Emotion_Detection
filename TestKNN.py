from PrepareData import X_train_combined, y_train_combined, X_test_combined, y_test_combined
from models.PCA_KNN import PCA_KNN
import numpy as np

# ฟังก์ชันการเปรียบเทียบโมเดล (เฉพาะ PCA + KNN)
def evaluate_pca_knn(X_train, y_train, X_test, y_test):
    model = PCA_KNN()

    # ฝึกโมเดล
    model.train(X_train, y_train)

    # ทดสอบโมเดล
    accuracy, train_time = model.evaluate(X_test, y_test)

    return accuracy, train_time

# แปลง y_train และ y_test เป็น 1D labels
if len(y_train_combined.shape) > 1:
    y_train_combined = np.argmax(y_train_combined, axis=1)

if len(y_test_combined.shape) > 1:
    y_test_combined = np.argmax(y_test_combined, axis=1)

# ประเมินผลลัพธ์ของ PCA + KNN
accuracy, train_time = evaluate_pca_knn(X_train_combined, y_train_combined, X_test_combined, y_test_combined)

# แสดงผลลัพธ์
print(f"PCA + KNN - Accuracy: {accuracy:.4f}, Time: {train_time:.4f} seconds")

