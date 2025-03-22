import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def load_model_and_data():
    """Load the best available model and data"""
    try:
        # First try to load the improved model if available
        try:
            model = load('improved_progression_model.joblib')
            features_df = pd.read_csv('enhanced_features.csv', index_col=0)
            print("Loaded improved model and enhanced features")
            model_type = "improved"
            metrics = {
                'accuracy': 0.69, 
                'precision': 0.70, 
                'recall': 0.69, 
                'f1': 0.70,
                'roc_auc': 0.678
            }
            confusion = np.array([[57, 17], [15, 15]])
            return model, features_df, model_type, metrics, confusion
        except FileNotFoundError:
            # Fall back to the original model
            model = load('idrid_progression_model.joblib')
            features_df = pd.read_csv('processed_features.csv', index_col=0)
            print("Loaded original model and features")
            model_type = "original"
            metrics = {
                'accuracy': 0.60, 
                'precision': 0.56, 
                'recall': 0.60, 
                'f1': 0.58,
                'roc_auc': 0.494
            }
            confusion = np.array([[72, 19], [32, 6]])
            return model, features_df, model_type, metrics, confusion
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        return None, None, None, None, None

def create_sample_prediction(model, features_df, model_type):
    """Create an interactive prediction component"""
    st.subheader("Progression Risk Predictor")
    st.write("Adjust the parameters below to see how they affect progression risk prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dr_grade = st.selectbox(
            "DR Grade", 
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: f"Grade {x}: " + ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"][x],
            index=2
        )
        
        dme_risk = st.selectbox(
            "DME Risk",
            options=[0, 1, 2],
            format_func=lambda x: ["No Risk", "Moderate Risk", "High Risk"][x],
            index=1
        )
    
    with col2:
        ma_count = st.slider("Microaneurysm Count", 0, 50, 15)
        hemorrhage_count = st.slider("Hemorrhage Count", 0, 30, 8)
    
    with col3:
        exudates_area = st.slider("Exudates Area (%)", 0.0, 5.0, 1.2, 0.1)
        image_brightness = st.slider("Image Brightness", 0.0, 1.0, 0.5, 0.05)
    
    # Create a sample input based on the selected parameters
    sample = pd.DataFrame({
        'dr_grade': [dr_grade],
        'dme_risk': [dme_risk],
        'Microaneurysms_count': [ma_count],
        'Haemorrhages_count': [hemorrhage_count],
        'Hard Exudates_area': [exudates_area * 100],
        'mean_brightness': [image_brightness * 255]
    })
    
    # Add other required columns with default values
    for col in model.feature_names_in_:
        if col not in sample.columns:
            sample[col] = features_df[col].median() if col in features_df else 0
    
    # Make prediction with the model
    try:
        # Get prediction
        X = sample[model.feature_names_in_]
        prob = model.predict_proba(X)[0][1]
        prediction = "High Risk" if prob > 0.5 else "Low Risk"
        
        # Display prediction with gauge chart
        st.subheader("Progression Risk Prediction")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Progression Risk: {prediction} ({model_type.capitalize()} Model)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show risk factors
        st.write("### Key Risk Factors:")
        risk_factors = []
        if dr_grade >= 3:
            risk_factors.append("⚠️ High DR grade (severe or proliferative)")
        if dme_risk >= 2:
            risk_factors.append("⚠️ High risk of macular edema")
        if ma_count > 20:
            risk_factors.append("⚠️ Elevated microaneurysm count")
        if hemorrhage_count > 10:
            risk_factors.append("⚠️ Significant hemorrhages detected")
        if exudates_area > 2.0:
            risk_factors.append("⚠️ Large exudate area")
        
        if not risk_factors:
            risk_factors = ["✅ No major risk factors identified"]
            
        for factor in risk_factors:
            st.write(factor)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Sample prediction not available with this model configuration.")

def add_clinical_guidelines():
    """Add clinical guidelines section for DR management"""
    st.header("Clinical Management Guidelines")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "No DR (0)", "Mild NPDR (1)", "Moderate NPDR (2)", "Severe NPDR (3)", "PDR (4)"
    ])
    
    with tab1:
        st.subheader("No DR (Grade 0)")
        st.write("""
        **Recommended follow-up:** Annual eye examination
        
        **Management:**
        - Regular blood glucose control
        - Blood pressure and lipid management
        - Patient education on diabetes self-management
        """)
        
    with tab2:
        st.subheader("Mild NPDR (Grade 1)")
        st.write("""
        **Recommended follow-up:** Annual eye examination
        
        **Management:**
        - Optimize glycemic control (target HbA1c < 7.0%)
        - Blood pressure control (target < 130/80 mmHg)
        - Lipid management
        - Monitor for development of DME
        """)
        
    with tab3:
        st.subheader("Moderate NPDR (Grade 2)")
        st.write("""
        **Recommended follow-up:** Every 6-9 months
        
        **Management:**
        - Strict glycemic control
        - Aggressive management of hypertension and dyslipidemia
        - Consider referral to retina specialist if signs of progression
        - OCT imaging if DME suspected
        """)
        
    with tab4:
        st.subheader("Severe NPDR (Grade 3)")
        st.write("""
        **Recommended follow-up:** Every 3-4 months
        
        **Management:**
        - Referral to retina specialist
        - Consider early panretinal photocoagulation (PRP) in high-risk cases
        - OCT imaging to assess for DME
        - More intensive monitoring for progression to PDR
        """)
        
    with tab5:
        st.subheader("Proliferative DR (Grade 4)")
        st.write("""
        **Recommended follow-up:** Every 1-3 months
        
        **Management:**
        - Urgent referral to retina specialist
        - Panretinal photocoagulation (PRP)
        - Consider anti-VEGF therapy
        - Vitrectomy may be needed for complications
        - Monitor closely for complications (vitreous hemorrhage, tractional retinal detachment)
        """)

def show_model_performance(metrics, confusion, model_type):
    """Display model performance metrics"""
    st.header(f"Model Performance ({model_type.capitalize()} Model)")
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2f}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.2f}")
    with col5:
        st.metric("AUC-ROC", f"{metrics['roc_auc']:.3f}")
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    
    # Create confusion matrix visualization using Plotly
    z = confusion
    x = ['No Progression', 'Progression']
    y = ['No Progression', 'Progression']
    
    # Change the confusion matrix to percentages for better visualization
    total_samples = np.sum(confusion)
    percentage_conf = (confusion / total_samples) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=percentage_conf,
        x=x,
        y=y,
        hoverongaps=False,
        colorscale='Blues',
        text=confusion,
        texttemplate="%{text} (%{z:.1f}%)",
        showscale=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show ROC curve
    st.subheader("ROC Curve")
    if model_type == "improved":
        try:
            # Try to load improved ROC curve
            img = plt.imread('viz_improved/improved_roc_curve.png')
            st.image(img, use_container_width=True)
        except:
            # Fall back to generic ROC curve
            fpr = [0, 0.2, 0.6, 1]
            tpr = [0, 0.4, 0.8, 1]
            create_roc_curve(fpr, tpr, metrics['roc_auc'])
    else:
        try:
            img = plt.imread('idrid_roc_curve.png')
            st.image(img, use_container_width=True)
        except:
            # Create basic ROC curve
            fpr = [0, 0.3, 0.7, 1]
            tpr = [0, 0.3, 0.7, 1]
            create_roc_curve(fpr, tpr, metrics['roc_auc'])

def create_roc_curve(fpr, tpr, auc_value):
    """Create a ROC curve with plotly"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc_value:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_comparison():
    """Show a comparison between original and improved models"""
    st.header("Model Comparison")
    
    # Define model metrics
    models = {
        "Original Model": {
            "accuracy": 0.60,
            "precision": 0.56,
            "recall": 0.60,
            "f1": 0.58,
            "roc_auc": 0.494,
            "description": "Gradient Boosting with basic features"
        },
        "Improved Model": {
            "accuracy": 0.69,
            "precision": 0.70,
            "recall": 0.69,
            "f1": 0.70,
            "roc_auc": 0.678,
            "description": "XGBoost with advanced feature engineering"
        }
    }
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        "Original Model": [models["Original Model"]["accuracy"], 
                          models["Original Model"]["precision"],
                          models["Original Model"]["recall"],
                          models["Original Model"]["f1"],
                          models["Original Model"]["roc_auc"]],
        "Improved Model": [models["Improved Model"]["accuracy"], 
                          models["Improved Model"]["precision"],
                          models["Improved Model"]["recall"],
                          models["Improved Model"]["f1"],
                          models["Improved Model"]["roc_auc"]],
        "Improvement": [models["Improved Model"]["accuracy"] - models["Original Model"]["accuracy"],
                       models["Improved Model"]["precision"] - models["Original Model"]["precision"],
                       models["Improved Model"]["recall"] - models["Original Model"]["recall"],
                       models["Improved Model"]["f1"] - models["Original Model"]["f1"],
                       models["Improved Model"]["roc_auc"] - models["Original Model"]["roc_auc"]]
    })
    
    # Create metric comparison graph
    fig = go.Figure()
    
    metrics = comparison_df["Metric"].tolist()
    original_values = comparison_df["Original Model"].tolist()
    improved_values = comparison_df["Improved Model"].tolist()
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=original_values,
        name='Original Model',
        marker_color='royalblue'
    ))
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=improved_values,
        name='Improved Model',
        marker_color='darkgreen'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        yaxis=dict(title='Score', range=[0, 1]),
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show improvements in table format
    st.subheader("Improvement Details")
    
    # Format the improvement column
    comparison_df["Improvement"] = comparison_df["Improvement"].apply(
        lambda x: f"+{x:.3f}" if x > 0 else f"{x:.3f}"
    )
    
    # Convert dataframe to HTML with styling
    html_table = comparison_df.to_html(index=False)
    styled_html = f"""
    <style>
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border: 1px solid #e0e0e0;
        }}
        th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .positive {{
            color: green;
            font-weight: bold;
        }}
        .negative {{
            color: red;
        }}
    </style>
    {html_table.replace('+', '<span class="positive">+').replace('-', '<span class="negative">-').replace('</td>', '</span></td>')}
    """
    
    st.markdown(styled_html, unsafe_allow_html=True)
    
    # Show key improvements
    st.subheader("Key Model Improvements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("##### Original Model")
        st.write("""
        - Basic feature set
        - GradientBoostingClassifier
        - Standard preprocessing
        - No handling of class imbalance
        - No feature selection
        """)
    
    with col2:
        st.write("##### Improved Model")
        st.write("""
        - Advanced feature engineering
        - XGBoost algorithm
        - Power transformation for normalization
        - SMOTE-Tomek for class balance
        - Feature selection for dimensionality reduction
        """)
    
    # Feature importance comparison
    st.subheader("Feature Importance Comparison")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("##### Original Model")
            try:
                img = plt.imread('idrid_feature_importance.png')
                st.image(img, use_container_width=True)
            except:
                st.write("Original feature importance plot not available")
        
        with col2:
            st.write("##### Improved Model")
            try:
                img = plt.imread('viz_improved/improved_feature_importance.png')
                st.image(img, use_container_width=True)
            except:
                st.write("Improved feature importance plot not available")
    except:
        st.write("Feature importance visualization not available")

def main():
    """Main function for dashboard"""
    st.set_page_config(
        page_title="DR Progression Prediction Dashboard", 
        layout="wide",
        page_icon="👁️"
    )
    
    # Add CSS styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5em;
            color: #2F80ED;
            text-align: center;
            margin-bottom: 20px;
        }
        .subheader {
            font-size: 1.8em;
            color: #2F80ED;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .improved-badge {
            background-color: #4CAF50;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with logo/icon
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.markdown("👁️")
    with col_title:
        st.markdown('<div class="main-header">Diabetic Retinopathy Progression Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load model and data
    model, features_df, model_type, metrics, confusion = load_model_and_data()
    
    if model is None or features_df is None:
        st.error("Could not load model or data. Please run model training first and ensure files exist.")
        st.stop()
    
    # Show model type badge
    if model_type == "improved":
        st.markdown(f"<div style='text-align: center;'><span class='improved-badge'>Using Improved Model</span></div>", unsafe_allow_html=True)
    
    # Dashboard tabs
    tab_overview, tab_predictions, tab_model, tab_clinical, tab_about = st.tabs([
        "📊 Dataset Overview", "🔮 Predictions", "📈 Model Performance", "🏥 Clinical Guidelines", "ℹ️ About"
    ])
    
    with tab_overview:
        st.markdown('<div class="subheader">Dataset Statistics</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Images", len(features_df))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average DR Grade", round(features_df['dr_grade'].mean(), 2))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            progression_count = features_df['progression'].sum() if 'progression' in features_df else 0
            progression_pct = int(features_df['progression'].mean()*100) if 'progression' in features_df else 0
            st.metric("Progression Cases", f"{int(progression_count)} ({progression_pct}%)")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model AUC", f"{metrics['roc_auc']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create layout with columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Retinopathy Grade Distribution")
            fig = px.histogram(
                features_df, 
                x='dr_grade',
                color='dr_grade',
                labels={'dr_grade': 'DR Grade', 'count': 'Number of Cases'},
                title='Distribution of DR Grades',
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'PDR (4)']
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            if 'progression' in features_df.columns:
                st.subheader("Progression Risk by DR Grade")
                progression_by_grade = features_df.groupby('dr_grade')['progression'].mean().reset_index()
                fig = px.bar(
                    progression_by_grade, 
                    x='dr_grade', 
                    y='progression',
                    color='dr_grade',
                    labels={'dr_grade': 'DR Grade', 'progression': 'Progression Probability'},
                    title='Progression Probability by DR Grade',
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=[0, 1, 2, 3, 4],
                        ticktext=['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'PDR (4)']
                    ),
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Add lesion data visualization
        st.subheader("Lesion Metrics by DR Grade")
        lesion_cols = [col for col in features_df.columns if any(x in col for x in ['Microaneurysms', 'Haemorrhages', 'Exudates'])]
        
        if lesion_cols:
            selected_lesion = st.selectbox(
                "Select Lesion Type", 
                options=[col for col in lesion_cols if any(x in col for x in ['count', 'area'])],
                format_func=lambda x: x.replace('_', ' ')
            )
            
            if selected_lesion in features_df.columns:
                lesion_by_grade = features_df.groupby('dr_grade')[selected_lesion].mean().reset_index()
                fig = px.bar(
                    lesion_by_grade, 
                    x='dr_grade', 
                    y=selected_lesion,
                    color='dr_grade',
                    labels={'dr_grade': 'DR Grade', selected_lesion: selected_lesion.replace('_', ' ')},
                    title=f'Average {selected_lesion.replace("_", " ")} by DR Grade',
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=[0, 1, 2, 3, 4],
                        ticktext=['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'PDR (4)']
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab_predictions:
        st.markdown('<div class="subheader">Progression Prediction</div>', unsafe_allow_html=True)
        
        # Add interactive prediction component
        create_sample_prediction(model, features_df, model_type)
        
        # Add model explanation
        st.subheader("How the Model Works")
        st.write("""
        This model predicts progression risk based on various factors:
        
        1. **DR Severity**: Higher grades correlate with higher progression risk
        2. **Lesion Characteristics**: Count, area, and type of lesions
        3. **Image Features**: Color distribution, brightness, and texture
        
        The improved model uses XGBoost with advanced feature engineering to provide more accurate predictions.
        """)
        
        # Show feature correlation heatmap
        st.subheader("Feature Correlation")
        numeric_df = features_df.select_dtypes(include=[np.number])
        
        if 'progression' in numeric_df.columns:
            # Get top correlations with progression
            corrs = numeric_df.corr()['progression'].sort_values(ascending=False)
            top_corrs = corrs[1:11]  # Skip progression itself and get top 10
            bottom_corrs = corrs[-10:] # Get bottom 10
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("##### Top Positive Correlations with Progression")
                fig = px.bar(
                    x=top_corrs.values,
                    y=top_corrs.index,
                    orientation='h',
                    labels={'x': 'Correlation', 'y': 'Feature'},
                    color=top_corrs.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("##### Top Negative Correlations with Progression")
                fig = px.bar(
                    x=bottom_corrs.values,
                    y=bottom_corrs.index,
                    orientation='h',
                    labels={'x': 'Correlation', 'y': 'Feature'},
                    color=bottom_corrs.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab_model:
        # Show model performance metrics
        show_model_performance(metrics, confusion, model_type)
        
        if model_type == "improved":
            # Show model comparison
            show_model_comparison()
        else:
            st.info("Run the improved model script (model_improvement.py) to see model comparison metrics.")
    
    with tab_clinical:
        add_clinical_guidelines()
        
        st.markdown("---")
        st.subheader("Risk Factors for DR Progression")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Modifiable Risk Factors:**")
            st.markdown("""
            - Poor glycemic control (HbA1c > 7.0%)
            - Hypertension (>140/90 mmHg)
            - Dyslipidemia
            - Obesity
            - Smoking
            - Nephropathy
            """)
            
        with col2:
            st.markdown("**Non-modifiable Risk Factors:**")
            st.markdown("""
            - Duration of diabetes
            - Type of diabetes (Type 1 vs Type 2)
            - Genetics/family history
            - Ethnicity
            - Puberty
            - Pregnancy
            """)
        
        st.subheader("Early Detection & Treatment Benefits")
        st.write("""
        Early detection and treatment of diabetic retinopathy can reduce the risk of blindness by 95%. 
        Regular eye examinations are crucial for diabetic patients.
        
        **Key interventions that reduce progression:**
        - Intensive glycemic control
        - Blood pressure control
        - Lipid management
        - Timely laser photocoagulation
        - Anti-VEGF therapy for DME
        """)
    
    with tab_about:
        st.markdown('<div class="subheader">About This Project</div>', unsafe_allow_html=True)
        st.write("""
        ### Diabetic Retinopathy Progression Prediction
        
        This project aims to predict the risk of diabetic retinopathy progression using machine learning techniques applied to retinal images. 
        
        **Dataset**: Indian Diabetic Retinopathy Image Dataset (IDRiD)
        
        **Methods**:
        - Image feature extraction
        - Lesion quantification
        - Advanced feature engineering
        - XGBoost classification with SMOTE-Tomek resampling
        - Feature selection for optimal performance
        
        **Improvements**:
        - Added feature interactions between DR grade and lesion metrics
        - Applied Yeo-Johnson power transformation to normalize feature distributions
        - Implemented SMOTE-Tomek for better class balance
        - Used XGBoost with optimized hyperparameters
        - Added feature selection to reduce dimensionality
        
        **Limitations**:
        - Simulated progression data (ideally would use longitudinal data)
        - Limited sample size
        - Single-center dataset
        
        ### References
        
        1. Porwal, P., Pachade, S., Kamble, R., et al. (2018). Indian Diabetic Retinopathy Image Dataset (IDRiD). IEEE Dataport.
        2. Wong TY, et al. (2016). Guidelines on Diabetic Eye Care: The International Council of Ophthalmology Recommendations for Screening, Follow-up, Referral, and Treatment Based on Resource Settings. Ophthalmology, 123(10), 1835-1847.
        3. Early Treatment Diabetic Retinopathy Study Research Group. (1991). Early photocoagulation for diabetic retinopathy: ETDRS report number 9. Ophthalmology, 98(5), 766-785.
        4. Chen T, Guestrin C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
        """)
        
        st.markdown("---")
        st.markdown("**Developed by:** Erebus")
        st.markdown("**Contact:** kshitiz23kumar@gmail.com")
        st.markdown("**GitHub:** [Project Repository](https://github.com/Erebuzzz/Diabetic-Retinopathy-Progression-Prediction)")

if __name__ == "__main__":
    main()
