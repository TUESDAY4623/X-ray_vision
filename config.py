"""
Configuration file for Multi-Disease X-Ray Detection System
"""
import os

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "X-ray_data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Primary model path (alias used by gui_app and model_inference)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "xray_model.h5")
MODEL_PATH      = MODEL_SAVE_PATH

# OpenVINO IR model directory
OPENVINO_MODEL_DIR = os.path.join(BASE_DIR, "models", "openvino")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ===== CLASSES =====
CLASSES = [
    'BONE_FRACTURE',
    'BONE_SUPPRESSION',
    'BRAIN_TUMOR',
    'CHEST_NIH_ABNORMAL',
    'COVID19',
    'NORMAL',
    'PNEUMONIA'
]
NUM_CLASSES = len(CLASSES)

# ===== IMAGE SETTINGS =====
IMG_SIZE        = 224          # ResNet50 input size
INPUT_SHAPE     = (IMG_SIZE, IMG_SIZE, 3)
IMG_CHANNELS    = 3
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# ===== TRAINING HYPERPARAMETERS =====
LEARNING_RATE           = 0.00005
EPOCHS                  = 30
BATCH_SIZE              = 48
EARLY_STOPPING_PATIENCE = 30
REDUCE_LR_PATIENCE      = 30
MIN_LEARNING_RATE       = 1e-7
REDUCE_LR_FACTOR        = 0.3

# ===== DATA AUGMENTATION =====
ROTATION_RANGE  = 25
HORIZONTAL_FLIP = True
VERTICAL_FLIP   = True
WIDTH_SHIFT_RANGE  = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ZOOM_RANGE      = 0.3
BRIGHTNESS_RANGE = [0.7, 1.3]
FILL_MODE       = 'nearest'

# ===== MODEL ARCHITECTURE =====
FINE_TUNE_LAYERS = 30
DROPOUT_1 = 0.5
DROPOUT_2 = 0.4
DENSE_1   = 1024
DENSE_2   = 512

# ===== HARDWARE ACCELERATION =====
GPU_DEVICE        = 1
NPU_DEVICE        = "NPU"
USE_NPU           = True
GPU_MEMORY_GROWTH = True
JIT_COMPILE       = True

# ===== DISPLAY =====
VERBOSE = 2  # 0=silent, 1=progress bar, 2=one line per epoch

# ===== MEDICAL RECOMMENDATIONS =====
MEDICAL_RECOMMENDATIONS = {
    'BONE_FRACTURE': (
        "A potential bone fracture has been detected.\n\n"
        "• Seek immediate medical attention from an orthopedic specialist.\n"
        "• Avoid putting weight or stress on the affected area.\n"
        "• Apply ice packs (wrapped in cloth) to reduce swelling.\n"
        "• Immobilize the limb if possible until professional help is available.\n"
        "• Further imaging (CT scan or MRI) may be required for confirmation."
    ),
    'BONE_SUPPRESSION': (
        "Bone suppression pattern detected in the image.\n\n"
        "• Consult a radiologist for detailed analysis.\n"
        "• This finding may indicate bone marrow changes or other conditions.\n"
        "• Follow-up with a specialist is strongly recommended.\n"
        "• Additional laboratory tests may be needed for diagnosis."
    ),
    'BRAIN_TUMOR': (
        "WARNING: Possible brain tumor indicator detected.\n\n"
        "• URGENTLY consult a neurologist or neurosurgeon.\n"
        "• Do NOT delay medical evaluation for this finding.\n"
        "• An MRI with contrast is typically required for confirmation.\n"
        "• Biopsy and further imaging studies will likely be necessary.\n"
        "• Treatment options depend heavily on tumor type, location, and grade."
    ),
    'CHEST_NIH_ABNORMAL': (
        "Chest abnormality consistent with NIH chest dataset patterns detected.\n\n"
        "• Consult a pulmonologist or internal medicine specialist.\n"
        "• A high-resolution CT scan of the chest is recommended.\n"
        "• Pulmonary function tests may be ordered.\n"
        "• Report any symptoms: shortness of breath, cough, chest pain.\n"
        "• Follow-up imaging in 3–6 months may be advised."
    ),
    'COVID19': (
        "Radiological patterns consistent with COVID-19 pneumonia detected.\n\n"
        "• Isolate immediately to prevent potential spread.\n"
        "• Contact your healthcare provider or local COVID-19 hotline.\n"
        "• Get an RT-PCR test for definitive confirmation.\n"
        "• Monitor oxygen saturation with a pulse oximeter.\n"
        "• Seek emergency care if SpO₂ drops below 94% or breathing worsens."
    ),
    'NORMAL': (
        "No significant radiological abnormalities detected.\n\n"
        "• The X-ray appears normal based on AI analysis.\n"
        "• Continue regular health check-ups as recommended.\n"
        "• Report any new or worsening symptoms to your doctor.\n"
        "• Maintain a healthy lifestyle: exercise, balanced diet, no smoking.\n"
        "• Annual health screenings are still recommended."
    ),
    'PNEUMONIA': (
        "Radiological findings suggestive of pneumonia detected.\n\n"
        "• Consult a physician immediately for antibiotic therapy evaluation.\n"
        "• Rest and stay well-hydrated.\n"
        "• Monitor temperature and oxygen levels closely.\n"
        "• Sputum culture may be ordered to identify the causative organism.\n"
        "• Hospitalization may be required for severe cases or vulnerable patients."
    ),
}