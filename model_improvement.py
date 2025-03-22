import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

def load_processed_data():
    """Load the processed features dataset"""
    try:
        features_df = pd.read_csv('processed_features.csv')
        print(f"Loaded dataset with {features_df.shape[0]} samples and {features_df.shape[1]} features")
        
        # Display some information about the dataset
        print("Column names:", features_df.columns.tolist())
        print("First 5 rows:")
        print(features_df.head())
        
        return features_df
    except FileNotFoundError:
        print("Error: processed_features.csv not found. Please run the data preparation script first.")
        print("Checking for alternative files...")
        
        # Try to find any CSV files in the directory
        csv_files = list(Path('.').glob('*.csv'))
        if csv_files:
            print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
            print(f"Attempting to load {csv_files[0].name} instead...")
            try:
                features_df = pd.read_csv(csv_files[0])
                print(f"Loaded {csv_files[0].name} with {features_df.shape[0]} samples and {features_df.shape[1]} features")
                return features_df
            except Exception as e:
                print(f"Failed to load alternative CSV: {e}")
        
        return None

def analyze_features(features_df):
    """Analyze features for better feature engineering"""
    print("Analyzing features for improvements...")
    
    # Filter to only include numeric columns for correlation
    numeric_df = features_df.select_dtypes(include=[np.number])
    print(f"Using {numeric_df.shape[1]} numeric features for correlation analysis")
    
    # Check if 'progression' is in numeric columns
    if 'progression' not in numeric_df.columns:
        print("Warning: 'progression' column is not numeric or not found in dataset")
        correlations = pd.Series(dtype=float)
        skewed_features = []
        return skewed_features, correlations
    
    # Check feature correlations with progression
    correlations = numeric_df.corr()['progression'].sort_values(ascending=False)
    print("\nTop 10 features correlated with progression:")
    print(correlations.head(10))
    print("\nBottom 10 features correlated with progression:")
    print(correlations.tail(10))
    
    # Check for class imbalance
    if 'progression' in features_df.columns:
        class_counts = features_df['progression'].value_counts()
        print("\nClass distribution:")
        print(class_counts)
        if len(class_counts) >= 2:
            imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
            print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    else:
        print("Warning: 'progression' target column not found in dataset")
    
    # Check feature distributions
    print("\nChecking feature distributions...")
    skewed_features = []
    for col in numeric_df.columns:
        if col not in ['progression', 'progression_prob', 'dr_grade', 'dme_risk']:
            skew = numeric_df[col].skew()
            if abs(skew) > 1:
                skewed_features.append((col, skew))
    
    print(f"Found {len(skewed_features)} highly skewed features (skew > 1)")
    
    # Create visualizations directory
    vis_dir = Path("viz_improved")
    vis_dir.mkdir(exist_ok=True)
    
    # Visualize feature distributions for top correlated features
    if not correlations.empty and len(correlations) > 5:
        top_features = correlations.head(6).index.tolist()
        plt.figure(figsize=(15, 10))
        i = 1
        for feature in top_features:
            if feature != 'progression' and feature in numeric_df.columns:
                plt.subplot(2, 3, i)
                if 'progression' in features_df.columns:
                    sns.histplot(data=features_df, x=feature, hue='progression', kde=True)
                else:
                    sns.histplot(data=features_df, x=feature, kde=True)
                plt.title(f"{feature} distribution")
                i += 1
        plt.tight_layout()
        plt.savefig(vis_dir / 'top_feature_distributions.png')
    else:
        print("Not enough numeric features for visualization")
    
    # Create feature interaction plots
    if 'dr_grade' in numeric_df.columns and 'progression' in numeric_df.columns:
        plt.figure(figsize=(10, 6))
        # Convert dr_grade to int if needed
        features_df['dr_grade'] = pd.to_numeric(features_df['dr_grade'], errors='coerce')
        features_df['dr_grade'] = features_df['dr_grade'].fillna(0).astype(int)
        
        crosstab = pd.crosstab(features_df['dr_grade'], features_df['progression'])
        if not crosstab.empty:
            crosstab_pct = crosstab.div(crosstab.sum(1), axis=0)
            sns.heatmap(crosstab_pct, annot=True, fmt=".2f", cmap="YlGnBu", 
                        cbar_kws={'label': 'Progression Probability'})
            plt.title('Progression Probability by DR Grade')
            plt.xlabel('Progression')
            plt.ylabel('DR Grade')
            plt.savefig(vis_dir / 'dr_grade_progression_heatmap.png')
    
    # Print data types for troubleshooting
    print("\nColumn data types:")
    for col, dtype in features_df.dtypes.items():
        print(f"{col}: {dtype}")
    
    return skewed_features, correlations

def engineer_features(features_df):
    """Create new features that might improve model performance"""
    print("Engineering new features...")
    
    # 1. Create interaction features
    if 'dr_grade' in features_df.columns and 'dme_risk' in features_df.columns:
        features_df['dr_dme_interaction'] = features_df['dr_grade'] * features_df['dme_risk']
        print("Created DR grade and DME risk interaction feature")
    
    # 2. Create lesion-based interaction features
    lesion_columns = [col for col in features_df.columns if any(x in col for x in 
                    ['Microaneurysms', 'Haemorrhages', 'Hard Exudates'])]
    
    count_columns = [col for col in lesion_columns if 'count' in col]
    area_columns = [col for col in lesion_columns if 'area' in col]
    
    if count_columns and 'dr_grade' in features_df.columns:
        features_df['total_lesion_count'] = features_df[count_columns].sum(axis=1)
        features_df['lesion_count_dr_ratio'] = (features_df['total_lesion_count'] / (features_df['dr_grade'] + 1))
        print("Created total lesion count and DR grade ratio features")
    
    if area_columns:
        features_df['total_lesion_area'] = features_df[area_columns].sum(axis=1)
        print("Created total lesion area feature")
    
    # 3. Create image-based advanced features
    color_features = [col for col in features_df.columns if any(x in col for x in 
                      ['mean_', 'std_', 'brightness', 'contrast'])]
    
    if color_features:
        # Color ratios (useful for detecting specific types of retinal changes)
        if 'mean_red' in features_df.columns and 'mean_green' in features_df.columns:
            features_df['red_green_ratio'] = features_df['mean_red'] / features_df['mean_green'].replace(0, 0.001)
            print("Created red-green ratio feature")
        
        if 'std_red' in features_df.columns and 'std_green' in features_df.columns and 'std_blue' in features_df.columns:
            features_df['color_variation'] = features_df[['std_red', 'std_green', 'std_blue']].mean(axis=1)
            print("Created color variation feature")
    
    # 4. Handle missing values if any
    missing_counts = features_df.isna().sum()
    if missing_counts.sum() > 0:
        print(f"Found {missing_counts.sum()} missing values across {(missing_counts > 0).sum()} columns")
        # Fill missing values with median or most common value depending on distribution
        for col in features_df.columns:
            if missing_counts[col] > 0:
                if features_df[col].dtype in ['int64', 'float64']:
                    features_df[col] = features_df[col].fillna(features_df[col].median())
                else:
                    features_df[col] = features_df[col].fillna(features_df[col].mode()[0])
    
    return features_df

def train_improved_model(features_df):
    """Train an improved model with advanced techniques"""
    print("Training improved model...")
    
    # Prepare features and target
    # First handle non-numeric columns
    print("Original columns:", features_df.columns.tolist())
    
    # Drop non-feature columns by name
    columns_to_drop = ['image_id', 'file_path', 'filename', 'id', 'Image name', 'path']
    drop_cols = [col for col in columns_to_drop if col in features_df.columns]
    
    if 'progression' not in features_df.columns:
        print("Warning: 'progression' column not found. Creating dummy progression column.")
        # For demonstration, create a dummy progression column based on dr_grade
        # In a real scenario, you should handle this differently
        if 'dr_grade' in features_df.columns:
            # Higher DR grades more likely to progress
            features_df['progression'] = (features_df['dr_grade'] >= 3).astype(int)
        else:
            # Random progression
            features_df['progression'] = np.random.randint(0, 2, size=len(features_df))
        
    if 'progression_prob' not in features_df.columns and 'progression' in features_df.columns:
        # Create a probability column for visualization if it doesn't exist
        features_df['progression_prob'] = features_df['progression'].astype(float)
    
    # Convert object columns to categorical then to codes
    for col in features_df.select_dtypes(include=['object']):
        try:
            features_df[col] = pd.Categorical(features_df[col]).codes
        except:
            # If conversion fails, add to drop list
            drop_cols.append(col)
    
    # Drop problematic columns and ensure target variable is kept separate
    X = features_df.drop(['progression'] + drop_cols, axis=1, errors='ignore')
    y = features_df['progression']
    
    # Drop any remaining non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        X = X.drop(non_numeric_cols, axis=1)
    
    # Print final features being used
    print(f"Final feature set: {X.columns.tolist()}")
    print(f"Features shape: {X.shape}")
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    # Create a feature selector based on importance
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
    
    # Create a more advanced pipeline with SMOTE-Tomek for handling imbalance
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('power_transform', PowerTransformer(method='yeo-johnson')),
        ('feature_selection', selector),
        ('sampling', SMOTETomek(random_state=42)),
        ('classifier', xgb.XGBClassifier(random_state=42))
    ])
    
    # Define improved parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 4, 5],
        'classifier__min_child_weight': [1, 3, 5],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__gamma': [0, 0.1, 0.2]
    }
    
    # Use stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search
    print("Starting hyperparameter tuning (this may take a while)...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate the model
    print("\nEvaluating improved model...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\nClassification Report:")
    print(report)
    print(f"ROC AUC Score: {roc_auc}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Create directory for visualizations
    vis_dir = Path("viz_improved")
    vis_dir.mkdir(exist_ok=True)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Improved Model ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / 'improved_roc_curve.png')
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Progression', 'Progression'],
                yticklabels=['No Progression', 'Progression'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(vis_dir / 'improved_confusion_matrix.png')
    
    # Get feature importance
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        # Get selected feature names
        selected_features = X.columns[best_model.named_steps['feature_selection'].get_support()]
        
        # Get feature importance
        importances = best_model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title('Improved Model Feature Importance')
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
        plt.tight_layout()
        plt.savefig(vis_dir / 'improved_feature_importance.png')
    
    # Save the model
    dump(best_model, 'improved_progression_model.joblib')
    print("Improved model saved as 'improved_progression_model.joblib'")
    
    # Save selected features for reference
    if hasattr(best_model.named_steps['feature_selection'], 'get_support'):
        selected_features = X.columns[best_model.named_steps['feature_selection'].get_support()]
        with open(vis_dir / 'selected_features.txt', 'w') as f:
            f.write("Selected features:\n")
            for feature in selected_features:
                f.write(f"- {feature}\n")
        print(f"Selected {len(selected_features)} out of {X.shape[1]} features")
        
    return best_model, X, y

def main():
    """Main function to run the model improvement process"""
    print("Starting model improvement process...")
    
    # Load processed data
    features_df = load_processed_data()
    if features_df is None:
        return
    
    # Analyze features
    skewed_features, correlations = analyze_features(features_df)
    
    # Engineer new features
    features_df = engineer_features(features_df)
    
    # Save the enhanced dataset
    features_df.to_csv('enhanced_features.csv')
    print("Enhanced features saved to 'enhanced_features.csv'")
    
    # Train improved model
    best_model, X, y = train_improved_model(features_df)
    
    print("\nModel improvement process complete.")
    print("Run the dashboard with the improved model using: streamlit run dashboard.py")

if __name__ == "__main__":
    main()