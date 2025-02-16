# ฝึกโมเดล CNN
history = model.fit(X_train_combined, y_train_combined, epochs=10, batch_size=32, validation_data=(X_test_combined, y_test_combined))
