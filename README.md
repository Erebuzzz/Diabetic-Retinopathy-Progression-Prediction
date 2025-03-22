# Diabetic Retinopathy Progression Prediction

![Dashboard Preview](https://raw.githubusercontent.com/Erebuzzz/Diabetic-Retinopathy-Progression-Prediction/main/viz_improved/improved_roc_curve.png)

## 🚀 Live Demo

[**View Live Dashboard**](https://diabetic-retinopathy-progression-prediction.streamlit.app/)

## Project Overview

This project develops a machine learning model to predict the risk of diabetic retinopathy (DR) progression using retinal images. It uses the Indian Diabetic Retinopathy Image Dataset (IDRiD) and extracts meaningful features from both the images and segmentation masks to predict disease progression.

## Key Features

- **Advanced Image Feature Extraction**: Extracts color, texture, and edge features from retinal images
- **Lesion Analysis**: Quantifies microaneurysms, hemorrhages, exudates, and other DR lesions
- **Progression Prediction**: Uses XGBoost with SMOTE-Tomek resampling to predict progression likelihood
- **Interactive Dashboard**: Visualize results and make predictions with our Streamlit dashboard
- **Model Comparison**: Compare performance between basic and advanced models

## Dataset

The IDRiD dataset contains retinal fundus photographs with ground truth labels for:
- DR grade (0-4)
- Risk of macular edema (0-2)
- Segmentation masks for different lesion types

## Model Performance

### Improved Model
- AUC-ROC: 0.678
- Accuracy: 69%
- Precision: 70%
- Recall: 69%
- F1 Score: 70%

### Original Model
- AUC-ROC: 0.494
- Accuracy: 60%
- Precision: 56%
- Recall: 60%
- F1 Score: 58%

## Installation and Usage

1. Clone this repository:
git clone https://github.com/kshitizagr/Diabetic-Retinopathy-Progression-Prediction.git cd Diabetic-Retinopathy-Progression-Prediction

2. Install dependencies: 
pip install -r requirements.txt

3. Run the model: 
python drp_prediction.py

4. Run the improved model:
python drp_prediction.py

5. View the dashboard locally: 
streamlit run dashboard.py

6. Or visit the [live dashboard](https://diabetic-retinopathy-progression-prediction.streamlit.app/)

## Clinical Relevance

Early prediction of DR progression can help clinicians:
- Identify high-risk patients who need more frequent monitoring
- Implement timely interventions to prevent vision loss
- Optimize resource allocation in healthcare settings

## Technical Details

- **Feature Engineering**: Created interaction features between DR grade and lesion metrics
- **Feature Transformation**: Applied Yeo-Johnson power transformation to normalize distributions
- **Class Imbalance**: Implemented SMOTE-Tomek for better class balance
- **Feature Selection**: Used model-based selection to reduce dimensionality
- **Hyperparameter Tuning**: Optimized model parameters using grid search with cross-validation

## Future Work

- Incorporate longitudinal data for true progression prediction
- Implement deep learning approaches for feature extraction
- Develop a comprehensive web application for clinical use
- Expand dataset with multi-center images for better generalization


## Contributors

- Erebuzzz

## Acknowledgments

- Indian Diabetic Retinopathy Image Dataset (IDRiD) for providing the dataset
- The Streamlit team for their excellent dashboard framework
