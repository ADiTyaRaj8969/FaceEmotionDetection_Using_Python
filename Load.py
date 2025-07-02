import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === SETTINGS ===
IMG_SIZE = 48
BATCH_SIZE = 64
DATA_PATH = 'Emotion_dataset' 

# === AUGMENTATION & SPLIT ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,          # 80% train, 20% validation
    horizontal_flip=True
)

# === TRAINING DATA ===
train_generator = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',        # grayscale for 48x48 emotion images
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# === VALIDATION DATA ===
val_generator = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
