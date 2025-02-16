import cv2
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import exposure
import time

class HOG_SVM:
    def __init__(self):
        self.model = SVC()

    def train(self, X_train, y_train):
        X_train_gray = []
        for img in X_train:
            print(f"Original Shape of image: {img.shape}")  # ตรวจสอบรูปภาพแต่ละตัว
            # ปรับขนาดภาพให้เป็น 48x48
            img_resized = cv2.resize(img, (48, 48))
            print(f"Resized Shape of image: {img_resized.shape}")  # ตรวจสอบขนาดหลังปรับ
            if len(img_resized.shape) == 3:  # หากเป็นภาพสี (BGR)
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            elif len(img_resized.shape) != 2:  # หากไม่ใช่ภาพ (ควรจะมี 2 หรือ 3 ช่องสี)
                raise ValueError("Invalid image shape. Expected 2D or 3D image.")
            X_train_gray.append(img_resized)

        X_train_hog = [hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False) for img in X_train_gray]
        X_train_hog = [exposure.rescale_intensity(hog_feat) for hog_feat in X_train_hog]

        start_time = time.time()
        self.model.fit(X_train_hog, y_train)
        self.train_time = time.time() - start_time

    def evaluate(self, X_test, y_test):
        X_test_gray = []
        for img in X_test:
            print(f"Original Shape of image: {img.shape}")  # ตรวจสอบรูปภาพแต่ละตัว
            # ปรับขนาดภาพให้เป็น 48x48
            img_resized = cv2.resize(img, (48, 48))
            print(f"Resized Shape of image: {img_resized.shape}")  # ตรวจสอบขนาดหลังปรับ
            if len(img_resized.shape) == 3:  # หากเป็นภาพสี (BGR)
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            elif len(img_resized.shape) != 2:  # หากไม่ใช่ภาพ (ควรจะมี 2 หรือ 3 ช่องสี)
                raise ValueError("Invalid image shape. Expected 2D or 3D image.")
            X_test_gray.append(img_resized)

        X_test_hog = [hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False) for img in X_test_gray]
        X_test_hog = [exposure.rescale_intensity(hog_feat) for hog_feat in X_test_hog]

        accuracy = self.model.score(X_test_hog, y_test)
        return accuracy, self.train_time
