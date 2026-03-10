import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Your 7 classes (6 diseases + Normal)
CLASSES = ['NORMAL', 'BRAIN_TUMOR', 'PNEUMONIA', 'COVID19', 
           'BONE_FRACTURE', 'BONE_SUPPRESSION', 'CHEST_NIH_ABNORMAL']
NUM_CLASSES = len(CLASSES)

def train_multi_disease_xray():
    """
    Train on ALL 6 datasets simultaneously
    """
    BASE_DIR = "./X-ray_data"
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'val'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    print(f"✅ Found {train_gen.samples} training images")
    print(f"✅ Found {val_gen.samples} validation images")
    print(f"📊 Classes: {train_gen.class_indices}")
    
    # Create model
    base_model = ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5),
        keras.callbacks.ModelCheckpoint('multi_xray_detector.h5', save_best_only=True)
    ]
    
    # Train!
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('final_multi_xray_model.h5')
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig('training_results.png')
    plt.show()
    
    return model

# Run training
if __name__ == "__main__":
    model = train_multi_disease_xray()
    print("✅ Multi-disease X-ray model trained!")
