import cv2
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

data_folder = '..\\Data_Set'  # ใช้เส้นทางสัมพัทธ์

# ตรวจสอบการมีอยู่ของโฟลเดอร์ย่อยที่เก็บข้อมูล
folders = ['angry', 'happy', 'sad', 'neutral']
for folder_name in folders:
    folder_path = os.path.join(data_folder, folder_name)
    if not os.path.exists(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
    else:
        print(f"Folder {folder_path} found.")

# ฟังก์ชันทำ Augmentation
def augment_image(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # พลิกซ้าย-ขวา 50% ของภาพ
        iaa.Affine(rotate=(-10, 10)),  # หมุนภาพ -10 ถึง 10 องศา
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # เพิ่ม noise
    ])
    return seq.augment_image(image)

def load_images_from_folder(folder, label, size=(48, 48), augment=False):
    images = []
    labels = []
    if not os.path.exists(folder):
        print(f"Error: The folder {folder} does not exist.")
        return images, labels  # คืนค่ารายการว่างหากโฟลเดอร์ไม่พบ

    # ตรวจสอบไฟล์ในโฟลเดอร์
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            # ตรวจสอบนามสกุลไฟล์ภาพที่รองรับ
            if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, size)
                    images.append(img_resized)
                    labels.append(label)

                    # ถ้าต้องการ Augmentation
                    if augment:
                        img_aug = augment_image(img_resized)
                        images.append(img_aug)
                        labels.append(label)

    return images, labels

# ปรับ path ให้ถูกต้อง
data_folder = '..\\Data_Set'  # โฟลเดอร์หลัก

angry_images, angry_labels = load_images_from_folder(os.path.join(data_folder, 'angry'), label=0, augment=True)
happy_images, happy_labels = load_images_from_folder(os.path.join(data_folder, 'happy'), label=1, augment=True)
sad_images, sad_labels = load_images_from_folder(os.path.join(data_folder, 'sad'), label=2, augment=True)
neutral_images, neutral_labels = load_images_from_folder(os.path.join(data_folder, 'neutral'), label=3, augment=True)

# รวมข้อมูล
X = angry_images + happy_images + sad_images + neutral_images
y = angry_labels + happy_labels + sad_labels + neutral_labels

# Normalize ข้อมูล
X_combined = np.array(X) / 255.0
X_combined = X_combined.reshape(-1, 48, 48, 1)

# One-Hot Encoding
y_combined_one_hot = to_categorical(y, num_classes=4)  # 4 อารมณ์

# แบ่งข้อมูลเป็น Train และ Test
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined_one_hot, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_combined.reshape(X_train_combined.shape[0], -1), np.argmax(y_train_combined, axis=1))


# ตรวจสอบการกระจายของ class ก่อนใช้ SMOTE
plt.hist(np.argmax(y_train_combined, axis=1), bins=4, alpha=0.5, label='Before SMOTE')
plt.title('Distribution of Emotions Before SMOTE')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.legend()
plt.show()

# ตรวจสอบการกระจายของ class หลังใช้ SMOTE
plt.hist(y_train_balanced, bins=4, alpha=0.5, label='After SMOTE')
plt.title('Distribution of Emotions After SMOTE')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.legend()
plt.show()

# ตรวจสอบการกระจายของ class หลังใช้ SMOTE
unique, counts = np.unique(y_train_balanced, return_counts=True)
print(f'Distribution of classes after SMOTE: {dict(zip(unique, counts))}')