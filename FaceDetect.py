import os
import cv2
import mediapipe as mp
import numpy as np
from PrepareData import X_train_combined, y_train_combined, X_test_combined, y_test_combined
from models.PCA_KNN import PCA_KNN

# ตั้งค่า Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ฟังก์ชันโหลดโมเดล ถ้าไม่มีให้ฝึกใหม่
def load_or_train_pca_knn():
    model_path = "pca_knn_model.pkl"
    if os.path.exists(model_path):
        print("📥 Loading existing PCA_KNN model...")
        return PCA_KNN.load_model(model_path)
    else:
        print("🚀 Training new PCA_KNN model...")
        model = PCA_KNN()
        y_train = np.argmax(y_train_combined, axis=1) if len(y_train_combined.shape) > 1 else y_train_combined
        model.train(X_train_combined, y_train)
        model.save_model(model_path)
        return model

# โหลดโมเดล
pca_knn_model = load_or_train_pca_knn()

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# กำหนด label ของอารมณ์
EMOTION_LABELS = ["Angry", "Sad", "Happy", "Neutral"]

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # แปลงภาพเป็น RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ตรวจจับใบหน้า
        results = face_detection.process(rgb_frame)

        # ถ้าพบใบหน้า
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # ครอบตัดใบหน้า
                face = frame[y:y+h_box, x:x+w_box]

                if face.shape[0] > 0 and face.shape[1] > 0:
                    # แปลงเป็น grayscale และปรับขนาด
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48))

                    # แสดงภาพที่ถูก resize
                    cv2.imshow("Resized Face", face_resized)

                    # แปลงเป็น numpy array และปรับรูปร่าง
                    face_input = face_resized.flatten().reshape(1, -1)

                    # ทำนายอารมณ์
                    predicted_label = pca_knn_model.predict(face_input)[0]
                    emotion_text = EMOTION_LABELS[predicted_label]

                    # แสดงผลการทำนาย
                    print(f"Predicted Label: {predicted_label} -> {emotion_text}")

                    # วาดกรอบรอบใบหน้า
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

                    # เพิ่มตำแหน่งของข้อความให้ห่างจากกรอบ
                    cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # แสดงผล
        cv2.imshow("Face & Emotion Detection", frame)

        # กด 'q' เพื่อปิด
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

