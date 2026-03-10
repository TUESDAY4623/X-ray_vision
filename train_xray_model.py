
"""
Training Script Optimized for INTEL ARC GPU
With TensorFlow 2.14.0 (Stable for Intel Arc)
AUTO-DETECTS VERSION & AUTO-FIXES IF NEEDED
"""

import subprocess
import sys
import os

print("=" * 80)
print("🚀 TENSORFLOW TRAINING + INTEL ARC GPU (UPDATED)")
print("=" * 80 + "\n")

# ===== CRITICAL: Check & Fix TensorFlow Version =====
print("STEP 0: Checking TensorFlow Version...")
print("-" * 80)

try:
    import tensorflow as tf
    tf_version_str = tf.__version__
    
    # Parse version
    version_parts = tf_version_str.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])
    
    print(f"Current: TensorFlow {tf_version_str}")
    
    # Check if version is problematic
    if (major > 2) or (major == 2 and minor >= 20):
        print(f"⚠️  Version {tf_version_str} has Intel Arc detection issues!")
        print("   Fixing by downgrading to TensorFlow 2.14.0...\n")
        
        print("⏳ Uninstalling TensorFlow {tf_version_str}...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "tensorflow", "-y", "-q"],
                            stderr=subprocess.DEVNULL)
        print("✅ Uninstalled\n")
        
        print("⏳ Installing TensorFlow 2.14.0 (stable for Intel Arc)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.14.0", "-q"],
                            timeout=300)
        print("✅ TensorFlow 2.14.0 installed\n")
        
        # Reload TensorFlow
        import importlib
        importlib.reload(tf)
        print(f"✅ Using: TensorFlow {tf.__version__}\n")
    else:
        print(f"✅ TensorFlow {tf_version_str} is compatible with Intel Arc\n")
        
except Exception as e:
    print(f"⚠️  Version check/fix issue: {e}")
    print("   Attempting to continue anyway...\n")

print()

# ===== STEP 1: Setup Intel Arc Environment =====
print("STEP 1: Configuring Intel Arc GPU...")
print("-" * 80)

# Set Intel Arc GPU environment variables
os.environ['INTEL_DEVICE_ID'] = '1'
os.environ['ONEAPI_DEVICE_SELECTOR'] = 'gpu'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print("✅ Intel Arc environment variables set")
print("   INTEL_DEVICE_ID=1")
print("   ONEAPI_DEVICE_SELECTOR=gpu")
print("   TF_FORCE_GPU_ALLOW_GROWTH=true\n")

# ===== STEP 2: Install Required Packages =====
print("STEP 2: Installing Required Packages...")
print("-" * 80)

packages_to_install = [
    ('numpy', 'numpy'),
    ('PIL', 'Pillow'),
    ('matplotlib', 'matplotlib'),
    ('sklearn', 'scikit-learn'),
]

for module_name, package_name in packages_to_install:
    try:
        __import__(module_name)
        print(f"   ✅ {package_name} already installed")
    except ImportError:
        print(f"   ⏳ Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
            print(f"   ✅ {package_name} installed")
        except:
            print(f"   ⚠️  {package_name} install failed (non-critical)")

print()

# ===== STEP 3: Check Intel Arc GPU Detection =====
print("STEP 3: Detecting Intel Arc GPU...")
print("-" * 80)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}\n")
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"📊 GPUs Available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu}")
    
    print(f"📊 CPUs Available: {len(cpus)}\n")
    
    # Determine device
    if len(gpus) > 0:
        print("✅ INTEL ARC GPU DETECTED!")
        print("   Using Intel Arc for training (5-10x faster than CPU)\n")
        DEVICE = '/GPU:0'
        DEVICE_NAME = "INTEL_ARC"
        USE_GPU = True
        JIT_COMPILE = False
    else:
        print("⚠️  Intel Arc GPU NOT detected")
        print("   Using CPU instead (slower)")
        print("   💡 If GPU should be here:")
        print("      1. Check Device Manager for Intel Arc GPU")
        print("      2. Update Intel Graphics Driver")
        print("      3. Restart computer after driver update\n")
        DEVICE = '/CPU:0'
        DEVICE_NAME = "CPU"
        USE_GPU = False
        JIT_COMPILE = False
        
except ImportError:
    print("❌ TensorFlow not installed!")
    print("   Run: pip install tensorflow==2.14.0")
    sys.exit(1)

# Enable memory growth for Arc GPU (has limited VRAM)
if len(gpus) > 0:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("   ✅ GPU memory growth enabled (for limited VRAM)")
        except RuntimeError as e:
            print(f"   ⚠️  Could not enable memory growth: {e}")

print("=" * 80 + "\n")

# ===== NOW IMPORT EVERYTHING =====
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import config

# ===== Model Creation =====
def create_model():
    """Create ResNet50-based model optimized for Intel Arc"""
    print("🏗️  Building ResNet50 model...")
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    
    # Fine-tuning: Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    trainable_count = sum([1 for l in base_model.layers if l.trainable])
    print(f"   ✅ Loaded ResNet50 ({trainable_count} trainable layers)")
    
    # Custom head with dropout for Arc
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile without JIT (Intel Arc doesn't support it well)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   ✅ Model: {model.count_params():,} parameters")
    print(f"   ✅ Device: {DEVICE_NAME}")
    
    return model


def create_data_generators():
    """Create data generators with Intel Arc optimizations"""
    print("\n📊 Creating data generators...")
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        shear_range=0.15,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # For Intel Arc with limited VRAM, use moderate batch size
    batch_size = config.BATCH_SIZE
    print(f"   ℹ️  Batch size: {batch_size}")
    if batch_size > 32 and USE_GPU:
        print(f"   ⚠️  Intel Arc may have limited VRAM")
        print(f"   💡 If OOM error occurs, reduce batch_size to 16 or 8")
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        interpolation='bilinear'
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        interpolation='bilinear'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        interpolation='bilinear'
    )
    
    print(f"   ✅ Train: {train_generator.samples} samples")
    print(f"   ✅ Val: {val_generator.samples} samples")
    print(f"   ✅ Test: {test_generator.samples} samples")
    
    return train_generator, val_generator, test_generator


def create_callbacks():
    """Create training callbacks"""
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]


def plot_training_history(history):
    """Plot training results"""
    print("\n📈 Generating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train', marker='o', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', marker='s', linewidth=2)
    axes[0].set_title(f'Accuracy ({DEVICE_NAME})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train', marker='o', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val', marker='s', linewidth=2)
    axes[1].set_title(f'Loss ({DEVICE_NAME})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'training_history_{DEVICE_NAME.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {filename}")
    plt.close()


def main():
    """Main training"""
    print("=" * 80)
    print(f"🏥 XRAY MODEL TRAINING ({DEVICE_NAME})")
    print("=" * 80)
    
    if not os.path.exists(config.TRAIN_DIR):
        print(f"\n❌ Training dir not found: {config.TRAIN_DIR}")
        return
    
    # Create model
    model = create_model()
    
    # Create data
    train_gen, val_gen, test_gen = create_data_generators()
    
    if train_gen.samples == 0 or val_gen.samples == 0:
        print("\n❌ Not enough data!")
        return
    
    # Callbacks
    callbacks = create_callbacks()
    
    # Train on device
    print(f"\n🚀 Training on {DEVICE_NAME}...")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Device: {DEVICE_NAME}")
    if USE_GPU:
        print("   ⚡ Using Intel Arc GPU (5-10x faster than CPU)")
    print("=" * 80)
    
    with tf.device(DEVICE):
        history = model.fit(
            train_gen,
            epochs=config.EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
            workers=14,                    
            use_multiprocessing=False,     
            max_queue_size=5           
        )
    
    # Save
    model.save(config.MODEL_PATH)
    print(f"\n✅ Model saved: {config.MODEL_PATH}")
    
    # Plot
    plot_training_history(history)
    
    # Evaluate
    print("\n📊 Final Results:")
    print("=" * 80)
    
    with tf.device(DEVICE):
        train_acc = model.evaluate(train_gen, verbose=0)[1]
        val_acc = model.evaluate(val_gen, verbose=0)[1]
        test_acc = model.evaluate(test_gen, verbose=0)[1]
    
    print(f"   Train Accuracy: {train_acc*100:.2f}%")
    print(f"   Val Accuracy: {val_acc*100:.2f}%")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print("=" * 80)
    print("✅ Done!")


if __name__ == "__main__":
    main()