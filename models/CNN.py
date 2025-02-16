from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from PrepareData import X_train_combined, y_train_combined, X_test_combined, y_test_combined

class CNN:
    def __init__(self):
        # สร้างโมเดล CNN
        self.model = Sequential()

        # Conv Layer 1
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())  # เพิ่ม BatchNormalization

        # Conv Layer 2
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())  # เพิ่ม BatchNormalization

        # Conv Layer 3
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())  # เพิ่ม BatchNormalization

        # Flatten และ Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(4, activation='softmax'))  # 4 class สำหรับ 4 อารมณ์

        # คอมไพล์โมเดล
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        # ฝึกโมเดล
        if X_val is not None and y_val is not None:
            history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
        else:
            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return history

    def evaluate(self, X_test, y_test):
        # ประเมินผล
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f'Accuracy on test data: {test_acc * 100:.2f}%')

# สร้างและฝึกโมเดล
cnn_model = CNN()
# Now proceed with model training
cnn_model.train(X_train_combined, y_train_combined, X_test_combined, y_test_combined, epochs=5, batch_size=32)
cnn_model.evaluate(X_test_combined, y_test_combined)
