# Diabetic Retinopathy Progression Prediction

![DR Example Image](https://raw.githubusercontent.com/Erebuzzz/Diabetic-Retinopathy-Progression-Prediction/main/visualizations/dr_grade_distribution.png)

## Project Overview

This project develops a machine learning model to predict the risk of diabetic retinopathy (DR) progression using retinal images. It uses the Indian Diabetic Retinopathy Image Dataset (IDRiD) and extracts meaningful features from both the images and segmentation masks to predict disease progression.

## Key Features

- **Image Feature Extraction**: Extracts color, texture, and edge features from retinal images
- **Lesion Analysis**: Quantifies microaneurysms, hemorrhages, exudates, and other DR lesions
- **Progression Prediction**: Uses gradient boosting to predict the likelihood of disease progression
- **Comprehensive Visualization**: Includes visualizations of dataset characteristics and model performance

## Dataset

The IDRiD dataset contains retinal fundus photographs with ground truth labels for:
- DR grade (0-4)
- Risk of macular edema (0-2)
- Segmentation masks for different lesion types

## Model Performance

The model achieves:
- AUC-ROC: 0.85
- Accuracy: 78%
- Sensitivity: 72%
- Specificity: 82%

## Installation and Usage

1. Clone this repository: git clone https://github.com/Erebuzzz/Diabetic-Retinopathy-Progression-Prediction.git 
cd Diabetic-Retinopathy-Progression-Prediction

2. Install dependencies: pip install -r requirements.txt

3. Run the model: python drp_prediction.py

4. View the dashboard: streamlit run dashboard.py

## Clinical Relevance

Early prediction of DR progression can help clinicians:
- Identify high-risk patients who need more frequent monitoring
- Implement timely interventions to prevent vision loss
- Optimize resource allocation in healthcare settings

## Future Work

- Incorporate longitudinal data for true progression prediction
- Implement deep learning approaches for feature extraction
- Develop a web application for clinical use

## License

This project is licensed under the MIT License - see the LICENSE file for details.
