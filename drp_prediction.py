import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import class_weight
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# DATASET PATHS
base_dir = Path("c:/Users/kshit/OneDrive/Documents/GitHub/Diabetic-Retinopathy-Progression-Prediction/idrid")
grading_dir = base_dir / "Disease Grading/Groundtruths"
segmentation_dir = base_dir / "Segmentation/All Segmentations"

# Check if the paths exist
"""if not BASE_DIR.exists():
    raise FileNotFoundError(f"Base directory {BASE_DIR} not found.")
if not GRADING_DIR.exists():
    raise FileNotFoundError(f"Grading directory {GRADING_DIR} not found.")
if not SEGMENTATION_DIR.exists():
    raise FileNotFoundError(f"Segmentation directory {SEGMENTATION_DIR} not found.")"""

# HELPER FUNCTIONS

def extract_image_features(image_path):
    """Extract features from a retinal image."""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize for consistency (optional)
    img = cv2.resize(img, (800, 800))
    
    # Extract basic features
    features = {}
    
    # Color channel statistics
    for i, channel in enumerate(['red', 'green', 'blue']):
        features[f'{channel}_mean'] = img[:,:,i].mean()
        features[f'{channel}_std'] = img[:,:,i].std()
        features[f'{channel}_max'] = img[:,:,i].max()
        features[f'{channel}_min'] = img[:,:,i].min()
    
    # Convert to different color spaces for additional features
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Grayscale features
    features['gray_mean'] = gray.mean()
    features['gray_std'] = gray.std()
    
    # HSV features (particularly useful for medical images)
    features['hue_mean'] = hsv[:,:,0].mean()
    features['saturation_mean'] = hsv[:,:,1].mean()
    features['value_mean'] = hsv[:,:,2].mean()
    features['saturation_std'] = hsv[:,:,1].std()
    
    # Basic texture features using GLCM
    # (This is simplified - a full implementation would use scikit-image for GLCM)
    features['contrast'] = np.std(gray)
    
    # Edge detection to quantify retinal changes
    edges = cv2.Canny(gray, 100, 200)
    features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    return features

def calculate_lesion_metrics(image_id, segmentation_dir):
    """Calculate metrics from segmentation masks."""
    metrics = {}
    
    # Define the lesion types and their corresponding directories
    lesion_types = {
        'microaneurysms': 'Microaneurysms',
        'hemorrhages': 'Haemorrhages',
        'hard_exudates': 'Hard Exudates',
        'soft_exudates': 'Soft Exudates',
        'optic_disc': 'Optic Disc'
    }
    
    for lesion, subdir in lesion_types.items():
        # Find the corresponding mask
        mask_pattern = segmentation_dir / 'Training Set' / subdir / f"{image_id}*"
        mask_paths = glob(str(mask_pattern))
        
        if mask_paths:
            # Read the mask
            mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
            
            # Calculate metrics
            if mask is not None:
                area = np.sum(mask > 0)
                metrics[f'{lesion}_area'] = area
                metrics[f'{lesion}_presence'] = 1 if area > 0 else 0
                
                # Calculate additional shape metrics if the lesion is present
                if area > 0:
                    # Number of connected components (crude estimate of count)
                    num_labels, labels = cv2.connectedComponents(mask)
                    metrics[f'{lesion}_count'] = num_labels - 1  # Subtract 1 for background
                    
                    # Average size of lesions
                    if num_labels > 1:
                        metrics[f'{lesion}_avg_size'] = area / (num_labels - 1)
                    else:
                        metrics[f'{lesion}_avg_size'] = 0
                    
                    # Compactness (perimeter^2 / area)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        total_perimeter = sum([cv2.arcLength(cnt, True) for cnt in contours])
                        metrics[f'{lesion}_compactness'] = (total_perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
                else:
                    metrics[f'{lesion}_count'] = 0
                    metrics[f'{lesion}_avg_size'] = 0
                    metrics[f'{lesion}_compactness'] = 0
            else:
                # No mask found, set default values
                metrics[f'{lesion}_area'] = 0
                metrics[f'{lesion}_presence'] = 0
                metrics[f'{lesion}_count'] = 0
                metrics[f'{lesion}_avg_size'] = 0
                metrics[f'{lesion}_compactness'] = 0
        else:
            # No mask found, set default values
            metrics[f'{lesion}_area'] = 0
            metrics[f'{lesion}_presence'] = 0
            metrics[f'{lesion}_count'] = 0
            metrics[f'{lesion}_avg_size'] = 0
            metrics[f'{lesion}_compactness'] = 0
            
    return metrics

def prepare_idrid_data(base_dir, grading_dir, segmentation_dir):
    """Prepare features and targets from the IDRiD dataset."""
    
    # Read the grading CSV files
    train_labels_path = grading_dir / 'IDRiD_Disease_Grading_Training_Labels.csv'
    test_labels_path = grading_dir / 'IDRiD_Disease_Grading_Testing_Labels.csv'
    
    if not train_labels_path.exists():
        raise FileNotFoundError(f"Training labels file {train_labels_path} not found.")
    if not test_labels_path.exists():
        raise FileNotFoundError(f"Testing labels file {test_labels_path} not found.")
    
    train_labels = pd.read_csv(train_labels_path)
    test_labels = pd.read_csv(test_labels_path)
    
    # Combine training and testing data for our purpose
    all_labels = pd.concat([train_labels, test_labels], ignore_index=True)
    
    # Convert image IDs to strings if they aren't already
    all_labels['Image name'] = all_labels['Image name'].astype(str)
    
    # Extract the image IDs
    image_ids = all_labels['Image name'].tolist()
    
    # Initialize a DataFrame to store all features
    all_features = []
    
    # Process each image
    print("Extracting features from images...")
    for image_id in tqdm(image_ids):
        # Find the image path (first check training, then testing)
        train_path = base_dir / 'Original Images' / 'Training Set' / f"{image_id}.jpg"
        test_path = base_dir / 'Original Images' / 'Testing Set' / f"{image_id}.jpg"
        
        if train_path.exists():
            image_path = train_path
        elif test_path.exists():
            image_path = test_path
        else:
            print(f"Warning: Image {image_id} not found, skipping")
            continue
        
        # Extract features from the image
        try:
            image_features = extract_image_features(image_path)
            
            # Get the corresponding row from labels
            label_row = all_labels[all_labels['Image name'] == image_id].iloc[0]
            
            # Add DR grade and Risk of DME
            image_features['dr_grade'] = label_row['Retinopathy grade']
            image_features['dme_risk'] = label_row['Risk of macular edema']
            
            # Add image ID
            image_features['image_id'] = image_id
            
            # Check if this image has segmentation masks
            segmentation_image_path = segmentation_dir / 'Training Set' / f"{image_id}.jpg"
            if segmentation_image_path.exists():
                # Calculate lesion metrics from segmentation masks
                lesion_metrics = calculate_lesion_metrics(image_id, segmentation_dir)
                # Add to image features
                image_features.update(lesion_metrics)
            else:
                # Set default values for lesion metrics
                for metric in ['area', 'presence', 'count', 'avg_size', 'compactness']:
                    for lesion in ['microaneurysms', 'hemorrhages', 'hard_exudates', 'soft_exudates', 'optic_disc']:
                        image_features[f'{lesion}_{metric}'] = 0
            
            # Add to the list of features
            all_features.append(image_features)
            
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    return features_df

def simulate_progression_data(features_df):
    """
    Simulate progression data using the DR grades.
    
    In a real clinical setting, this would be replaced with actual
    follow-up data indicating whether DR progressed.
    """
    # Create a synthetic progression target
    # Higher DR grade = higher chance of progression
    np.random.seed(42)
    
    # Calculate a progression probability based on:
    # 1. Current DR grade
    # 2. DME risk
    # 3. Presence of lesions
    features_df['progression_prob'] = (
        features_df['dr_grade'] / 4.0 * 0.4 +  # DR grade contribution
        features_df['dme_risk'] / 2.0 * 0.2 +   # DME risk contribution
        (features_df['microaneurysms_presence'] * 0.1) +
        (features_df['hemorrhages_presence'] * 0.1) +
        (features_df['hard_exudates_presence'] * 0.1) +
        (features_df['soft_exudates_presence'] * 0.1)
    )
    
    # Add random noise
    features_df['progression_prob'] += np.random.normal(0, 0.1, size=len(features_df))
    
    # Clip probabilities to [0, 1]
    features_df['progression_prob'] = np.clip(features_df['progression_prob'], 0, 1)
    
    # Generate binary progression outcome
    features_df['progression'] = (np.random.random(size=len(features_df)) < features_df['progression_prob']).astype(int)
    
    return features_df

# ----------------- MAIN EXECUTION -----------------

def train_progression_model():
    """Train and evaluate the DR progression model."""
    
    # 1. Prepare the dataset
    print("Preparing the IDRiD dataset...")
    features_df = prepare_idrid_data(base_dir, grading_dir, segmentation_dir)
    
    # 2. Simulate progression data (in a real implementation, you'd use actual follow-up data)
    print("Simulating progression data...")
    features_df = simulate_progression_data(features_df)
    
    # 3. Split features and target
    # Exclude ID, probability, and target columns from features
    X = features_df.drop(['image_id', 'progression_prob', 'progression', 'dr_grade', 'dme_risk'], axis=1)
    y = features_df['progression']
    
    # 4. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 5. Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    # 6. Define parameter grid for grid search
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 4],
        'classifier__min_samples_split': [5, 10],
        'classifier__subsample': [0.8, 1.0]
    }
    
    # 7. Handle class imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Update the classifier with class weights
    pipeline.steps[-1] = ('classifier', GradientBoostingClassifier(
        random_state=42, 
        class_weight=class_weight_dict
    ))
    
    # 8. Perform grid search with cross-validation
    print("Performing grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # 9. Train the model
    grid_search.fit(X_train, y_train)
    
    # 10. Get the best model
    best_model = grid_search.best_estimator_
    
    # 11. Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # 12. Evaluate the model
    print("\nBest parameters:", grid_search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
    
    # 13. Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['No Progression', 'Progression'], 
               yticklabels=['No Progression', 'Progression'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('idrid_confusion_matrix.png')
    
    # 14. Feature importance
    classifier = best_model.named_steps['classifier']
    feature_importances = classifier.feature_importances_
    feature_names = X.columns
    
    # Sort feature importances
    sorted_idx = np.argsort(feature_importances)[::-1]
    
    # Plot top 15 feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(min(15, len(sorted_idx))), 
            feature_importances[sorted_idx][:15], 
            align='center')
    plt.xticks(range(min(15, len(sorted_idx))), 
              [feature_names[i] for i in sorted_idx][:15], 
              rotation=90)
    plt.tight_layout()
    plt.savefig('idrid_feature_importance.png')
    
    # 15. ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('idrid_roc_curve.png')
    
    # 16. Save the model
    from joblib import dump
    dump(best_model, 'idrid_progression_model.joblib')
    
    print("\nModel training and evaluation complete.")
    print("Model saved as 'idrid_progression_model.joblib'")
    
    return best_model, X, y

if __name__ == "__main__":
    # Make sure to update the dataset paths at the top of this file
    if not base_dir.exists():
        print(f"Error: Dataset directory {base_dir} not found.")
        print("Please update the BASE_DIR variable with the correct path.")
    else:
        try:
            model, X, y = train_progression_model()
        except Exception as e:
            print(f"An error occurred: {str(e)}")