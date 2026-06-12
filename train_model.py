import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================
# SETTING
# ==========================
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10

# ==========================
# DATA AUGMENTATION
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==========================
# LOAD DATASET
# ==========================
train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Tampilkan nama class
print("Class yang ditemukan:")
print(train_data.class_indices)

# ==========================
# MODEL CNN
# ==========================
model = models.Sequential([

    layers.Input(shape=(64, 64, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    # otomatis sesuai jumlah kelas
    layers.Dense(train_data.num_classes,
                 activation='softmax')
])

# ==========================
# COMPILE MODEL
# ==========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# TAMPILKAN MODEL
# ==========================
model.summary()

# ==========================
# TRAINING
# ==========================
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# ==========================
# SAVE MODEL
# ==========================
model.save("fruit_model.h5")

print("\nModel berhasil disimpan!")
print("Nama file: fruit_model.h5")

# ==========================
# EVALUASI MODEL
# ==========================
loss, accuracy = model.evaluate(test_data)

print(f"\nAkurasi Model: {accuracy * 100:.2f}%")

# ==========================
# GRAFIK HASIL TRAINING
# ==========================
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.show()
