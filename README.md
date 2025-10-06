# BrainTumorClassification

## Brain Tumor Classification from Medical Imaging

### 4th Year Project @ESME

BrainTumorClassification is a project that aims to classify brain tumors through medical image analysis. This project implements multiple Deep Learning approaches to classify brain MRI images into four categorie s: glioma, meningioma, pituitary, and no tumor using advanced convolutional neural networks and transfer learning techniques.

## Getting Started

### Prerequisites

- **Python** : 3.8+ supported
- **Jupyter Notebook**

### Required Libraries

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow
```

### Usage

1. **Clone the repository** :

   ```bash
   git clone https://github.com/siyamr/BrainTumorClassification.git
   cd BrainTumorClassification/Brain Detection
   ```

2. **Choose your approach** :
   - Open `ApprocheCNN.ipynb` for custom CNN approach
   - Open `ApprocheDenseNet121.ipynb` for DenseNet121 transfer learning
   - Open `ApprocheResNet50.ipynb` for ResNet50 transfer learning
   - Open `ApprocheVGG16.ipynb` for VGG16 transfer learning
   - Open `ApprocheVGG19.ipynb` for VGG19 transfer learning

3. **Run the notebooks** :
   - Execute cells sequentially to train and evaluate models
   - Each notebook is self-contained with data loading and preprocessing
   - A GPU is recommended for faster training

## Dataset

The brain MRI images used here include four categories of brain scans :

- **glioma_tumor/** : Glioma tumor MRI images
- **meningioma_tumor/** : Meningioma tumor MRI images
- **pituitary_tumor/** : Pituitary tumor MRI images
- **no_tumor/** : Healthy brain MRI images (no tumor)

The dataset is separated in ``Training` and `Testing` directories.

## Project Objectives

This project explores five distinct deep learning approaches for brain tumor classification :

### 1. Custom CNN Approach (`ApprocheCNN.ipynb`)

- **Architecture** : Custom Convolutional Neural Network
- **Methods** : From-scratch CNN design with multiple conv layers
- **Focus** : Custom feature extraction and pattern recognition

### 2. DenseNet121 Transfer Learning (`ApprocheDenseNet121.ipynb`)

- **Architecture** : Pre-trained DenseNet121
- **Methods** : Transfer learning with fine-tuning
- **Focus** : Dense connectivity and feature reuse

### 3. ResNet50 Transfer Learning (`ApprocheResNet50.ipynb`)

- **Architecture** : Pre-trained ResNet50
- **Methods** : Residual learning with transfer learning
- **Focus** : Deep residual networks and skip connections

### 4. VGG16 Transfer Learning (`ApprocheVGG16.ipynb`)

- **Architecture** : Pre-trained VGG16
- **Methods** : Classical CNN architecture with transfer learning
- **Focus** : Deep convolutional layers with small filters

### 5. VGG19 Transfer Learning (`ApprocheVGG19.ipynb`)

- **Architecture** : Pre-trained VGG19
- **Methods** : Extended VGG architecture with transfer learning
- **Focus** : Deeper network with enhanced feature extraction

## Project Structure

```text
BrainTumorClassification/
├── README
└── Brain Detection/
    ├── ApprocheCNN.ipynb          # Custom CNN approach
    ├── ApprocheDenseNet121.ipynb  # DenseNet121
    ├── ApprocheResNet50.ipynb     # ResNet50
    ├── ApprocheVGG16.ipynb        # VGG16
    ├── ApprocheVGG19.ipynb        # VGG19
    └── Brain-Tumor-Classification/
        ├── Training/              # Training dataset
        │   ├── glioma_tumor/
        │   ├── meningioma_tumor/
        │   ├── pituitary_tumor/
        │   └── no_tumor/
        └── Testing/               # Testing dataset
            ├── glioma_tumor/
            ├── meningioma_tumor/
            ├── pituitary_tumor/
            └── no_tumor/
```

## Features

- **Multi-Architecture Analysis** : Five complementary deep learning approaches
- **Transfer Learning** : Leveraging pre-trained models for enhanced performance
- **Medical Image Classification** : Specialized for brain MRI tumor detection
- **Comprehensive Evaluation** : Performance comparison across different architectures
- **Visualization** : Model performance analysis and prediction visualization

## Research Applications

This project contributes to :

- **Medical Diagnosis** : AI-assisted brain tumor detection and classification
- **Comparative Study** : Evaluation of different CNN architectures for medical imaging
- **Transfer Learning** : Application of pre-trained models in medical AI
- **Healthcare AI** : Advancement of AI in medical image analysis

## Contributors

- **Siyam R.** - ESME 2025 - AI Software Engineer
- **Lucile D.** - ESME 2025
- **Léo C.** - ESME 2025

## License

This project is part of an academic research initiative at ESME Sudria.

---

**Note** : This project is for educational purposes only.
