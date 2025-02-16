from PrepareData import X_train_combined, y_train_combined, X_test_combined, y_test_combined
from models.HOG_SVM import HOG_SVM
from models.PCA_KNN import PCA_KNN
from models.LBP_DecisionTree import LBP_DecisionTree
# ลบ CNN ออก
# from models.CNN import CNN
from keras.utils import to_categorical

import numpy as np
from sklearn.model_selection import train_test_split

# ฟังก์ชันการเปรียบเทียบโมเดล
def compare_models(X_train, y_train, X_test, y_test):
    models = {
        'HOG + SVM': HOG_SVM(),
        'PCA + KNN': PCA_KNN(),
        'LBP + Decision Tree': LBP_DecisionTree(),
        # ลบ CNN ออก
        # 'CNN': CNN()
    }

    results = {}

    # แบ่งข้อมูล training เป็น train และ validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # ใน compare_models
    for model_name, model in models.items():
        if model_name == 'CNN':
            model.train(X_train, y_train, X_val, y_val)  # ส่ง X_val, y_val เฉพาะ CNN
        else:
            # แปลง y_train_combined_onehot ให้เป็น 1D labels สำหรับโมเดลที่ไม่ใช่ CNN
            y_train_to_use = np.argmax(y_train, axis=1)
            model.train(X_train, y_train_to_use)  # ส่งแค่ X_train, y_train สำหรับโมเดลอื่นๆ
        # แปลง y_test_combined_onehot กลับเป็น 1D labels สำหรับโมเดลที่ไม่ใช่ CNN
        if model_name != 'CNN':
            y_test_to_use = np.argmax(y_test, axis=1)
        else:
            y_test_to_use = y_test  # สำหรับ CNN ใช้ one-hot encoding ตามเดิม
        accuracy, train_time = model.evaluate(X_test, y_test_to_use)
        results[model_name] = {
            'accuracy': accuracy,
            'time': train_time
        }

    return results

# แปลง y_train_combined และ y_test_combined กลับเป็น 1D labels
y_train_combined_int = np.argmax(y_train_combined, axis=1)
y_test_combined_int = np.argmax(y_test_combined, axis=1)

# แปลง y_train_combined_int และ y_test_combined_int กลับเป็น one-hot encoding
y_train_combined_onehot = to_categorical(y_train_combined_int, num_classes=4)
y_test_combined_onehot = to_categorical(y_test_combined_int, num_classes=4)

# ส่งข้อมูลให้ compare_models โดยใช้ one-hot สำหรับโมเดลที่ใช้ CNN
results = compare_models(X_train_combined, y_train_combined_onehot, X_test_combined, y_test_combined_onehot)

# แสดงผลลัพธ์
for model_name, result in results.items():
    print(f"{model_name} - Accuracy: {result['accuracy']:.4f}, Time: {result['time']:.4f} seconds")
