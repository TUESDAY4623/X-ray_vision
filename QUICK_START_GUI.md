# 🚀 Quick Start Guide - Tkinter GUI with Intel NPU

Quick guide to get the GUI running with Intel NPU support.

## ⚡ Fast Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train Model (if not already done)

```bash
python train_xray_model.py
```

### Step 3: Convert to OpenVINO (for NPU)

```bash
python convert_to_openvino.py
```

### Step 4: Configure NPU

Edit `config.py`:
```python
USE_NPU = True
NPU_DEVICE = 'NPU'
```

### Step 5: Run GUI

```bash
python gui_app.py
```

## 🎯 That's It!

The GUI will open and you can:
1. Click "Upload X-Ray Image"
2. Select an X-ray image
3. Click "Analyze X-Ray"
4. View results in tabs

## 📋 What You Get

✅ **Modern Tkinter GUI** - Professional desktop application  
✅ **Intel NPU Support** - 2-5x faster inference  
✅ **Grad-CAM Visualization** - See where model focuses  
✅ **Probability Charts** - Visual probability distribution  
✅ **Medical Recommendations** - Contextual guidance  

## 🔧 Troubleshooting

**NPU not detected?**
- System auto-falls back to CPU
- Still works, just slower

**Model not found?**
- Run: `python train_xray_model.py`
- Or use pre-trained model

**GUI won't start?**
- Check: `python -c "import tkinter"`
- Install: `sudo apt-get install python3-tk` (Linux)

## 📚 More Info

- Full guide: `GUI_USAGE_GUIDE.md`
- README: `README.md`

---

**Happy detecting! 🏥**
