┌─────────────────────────────────────────────────┐
│         Medical Image Upload                    │
│         (X-ray, MRI, CT Scan)                   │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│     Image Preprocessing                         │
│   • Resize to 224×224                           │
│   • Normalization                               │
│   • Data Augmentation                           │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│    Multi-Modal CNN Architecture                 │
│   ┌─────────────────────────────────────┐       │
│   │  ResNet50 / DenseNet121 / VGG16     │       │
│   │  (Transfer Learning)                │       │
│   │  Feature Extraction                 │       │
│   └────────┬────────────────────────────┘       │
│            │ Features                           │
│   ┌────────▼────────────────────────────┐       │
│   │  Feature Fusion Layer               │       │
│   │  (Concatenate features from all)    │       │
│   └────────┬────────────────────────────┘       │
│            │ Fused Features                     │
│   ┌────────▼────────────────────────────┐       │
│   │  Classification Head                │       │
│   │  • Dense layers                     │       │
│   │  • Softmax output                   │       │
│   └────────┬────────────────────────────┘       │
└────────────┼────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│         Model Output                            │
│  ┌─────────────────────────────────────┐        │
│  │ Disease: Pneumonia (92% confidence) │        │
│  │ Location: Right Lower Lobe          │        │
│  │ Severity: Moderate                  │        │
│  │ Recommendation: Hospitalization     │        │
│  │ Visual Explanation: [Heatmap]       │        │
│  └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────┘

