import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import ast
import json
import os

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the datasets from the same folder as the script
def load_datasets():
    """Load both training datasets from scripts folder"""
    train_df = pd.read_csv('train.csv')
    localizers_df = pd.read_csv('train_localizers.csv')
    return train_df, localizers_df

# Data exploration functions
def explore_train_data(df):
    """Explore the main training dataset"""
    print("=== TRAIN DATASET EXPLORATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    # Patient demographics
    print(f"\nPatient Age distribution:")
    print(df['PatientAge'].describe())
    
    print(f"\nPatient Sex distribution:")
    print(df['PatientSex'].value_counts())
    
    print(f"\nModality distribution:")
    print(df['Modality'].value_counts())
    
    # Aneurysm locations (binary columns)
    aneurysm_columns = [col for col in df.columns if col not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
    
    print(f"\nAneurysm location frequencies:")
    for col in aneurysm_columns:
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        print(f"{col}: {count} ({percentage:.2f}%)")
    
    # Overall aneurysm presence
    print(f"\nAneurysm Present distribution:")
    print(df['Aneurysm Present'].value_counts())
    print(f"Aneurysm prevalence: {df['Aneurysm Present'].mean():.3f}")
    
    return aneurysm_columns

def explore_localizers_data(df):
    """Explore the localizers dataset"""
    print("\n=== LOCALIZERS DATASET EXPLORATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Parse coordinates
    def parse_coordinates(coord_str):
        try:
            coord_dict = ast.literal_eval(coord_str)
            return coord_dict['x'], coord_dict['y']
        except:
            return None, None
    
    df['x_coord'] = df['coordinates'].apply(lambda x: parse_coordinates(x)[0])
    df['y_coord'] = df['coordinates'].apply(lambda x: parse_coordinates(x)[1])
    
    print(f"\nLocation distribution:")
    print(df['location'].value_counts())
    
    print(f"\nCoordinate statistics:")
    print(f"X coordinates: min={df['x_coord'].min():.2f}, max={df['x_coord'].max():.2f}, mean={df['x_coord'].mean():.2f}")
    print(f"Y coordinates: min={df['y_coord'].min():.2f}, max={df['y_coord'].max():.2f}, mean={df['y_coord'].mean():.2f}")
    
    return df

# Data preprocessing functions
def preprocess_train_data(df):
    """Preprocess the training dataset"""
    processed_df = df.copy()
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    processed_df['PatientSex_encoded'] = le_sex.fit_transform(processed_df['PatientSex'])
    
    le_modality = LabelEncoder()
    processed_df['Modality_encoded'] = le_modality.fit_transform(processed_df['Modality'])
    
    # Normalize age
    processed_df['PatientAge_normalized'] = (processed_df['PatientAge'] - processed_df['PatientAge'].mean()) / processed_df['PatientAge'].std()
    
    # Create feature matrix and target
    feature_columns = [col for col in processed_df.columns if col not in ['SeriesInstanceUID', 'PatientSex', 'Modality', 'Aneurysm Present']]
    X = processed_df[feature_columns]
    y = processed_df['Aneurysm Present']
    
    return X, y, le_sex, le_modality

def create_location_features(train_df, localizers_df):
    """Create features based on localizer data"""
    # Count aneurysms per series
    aneurysm_counts = localizers_df.groupby('SeriesInstanceUID').size().reset_index(name='aneurysm_count')
    
    # Get location diversity (number of unique locations per series)
    location_diversity = localizers_df.groupby('SeriesInstanceUID')['location'].nunique().reset_index(name='location_diversity')
    
    # Merge with train data
    enhanced_df = train_df.merge(aneurysm_counts, on='SeriesInstanceUID', how='left')
    enhanced_df = enhanced_df.merge(location_diversity, on='SeriesInstanceUID', how='left')
    
    # Fill NaN values (cases without localizer data)
    enhanced_df['aneurysm_count'] = enhanced_df['aneurysm_count'].fillna(0)
    enhanced_df['location_diversity'] = enhanced_df['location_diversity'].fillna(0)
    
    return enhanced_df

# Visualization functions
def create_visualizations(train_df, localizers_df):
    """Create various visualizations"""
    
    plt.figure(figsize=(15, 10))
    
    # 1. Age distribution by aneurysm presence
    plt.subplot(2, 3, 1)
    sns.boxplot(data=train_df, x='Aneurysm Present', y='PatientAge')
    plt.title('Age Distribution by Aneurysm Presence')
    
    # 2. Sex distribution by aneurysm presence
    plt.subplot(2, 3, 2)
    sex_aneurysm = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'], normalize='index')
    sex_aneurysm.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Sex Distribution by Aneurysm Presence')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    
    # 3. Modality distribution by aneurysm presence
    plt.subplot(2, 3, 3)
    modality_aneurysm = pd.crosstab(train_df['Modality'], train_df['Aneurysm Present'], normalize='index')
    modality_aneurysm.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Modality Distribution by Aneurysm Presence')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    
    # 4. Aneurysm location heatmap
    plt.subplot(2, 3, 4)
    aneurysm_columns = [col for col in train_df.columns if col not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
    location_correlations = train_df[aneurysm_columns].corr()
    sns.heatmap(location_correlations, annot=False, cmap='coolwarm', center=0, cbar=True)
    plt.title('Aneurysm Location Correlations')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # 5. Aneurysm location frequencies
    plt.subplot(2, 3, 5)
    location_counts = train_df[aneurysm_columns].sum().sort_values(ascending=True)
    location_counts.plot(kind='barh', ax=plt.gca())
    plt.title('Aneurysm Location Frequencies')
    plt.xlabel('Count')
    
    # 6. Coordinate scatter plot from localizers
    plt.subplot(2, 3, 6)
    if len(localizers_df) > 0:
        # Parse coordinates for plotting
        def parse_coordinates(coord_str):
            try:
                coord_dict = ast.literal_eval(coord_str)
                return coord_dict['x'], coord_dict['y']
            except:
                return None, None
        
        coords = localizers_df['coordinates'].apply(parse_coordinates)
        x_coords = [c[0] for c in coords if c[0] is not None]
        y_coords = [c[1] for c in coords if c[1] is not None]
        
        if x_coords and y_coords:
            plt.scatter(x_coords, y_coords, alpha=0.6)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Aneurysm Coordinates Distribution')
    
    plt.tight_layout()
    plt.show()

# Model preparation functions
def prepare_ml_features(train_df, localizers_df):
    """Prepare features for machine learning"""
    
    # Create enhanced dataset with localizer features
    enhanced_df = create_location_features(train_df, localizers_df)
    
    # Prepare features
    X, y, le_sex, le_modality = preprocess_train_data(enhanced_df)
    
    return X, y, le_sex, le_modality

# Simple machine learning model
def train_simple_model(X, y):
    """Train a simple Random Forest model"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Print results
    print("\n=== MODEL PERFORMANCE ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return rf_model, scaler, feature_importance

# Main analysis pipeline
def main_analysis():
    """Run the complete analysis pipeline"""
    
    # Load data
    print("Loading datasets...")
    train_df, localizers_df = load_datasets()
    
    # Explore data
    aneurysm_columns = explore_train_data(train_df)
    localizers_enhanced = explore_localizers_data(localizers_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(train_df, localizers_enhanced)
    
    # Prepare ML features
    print("\nPreparing machine learning features...")
    X, y, le_sex, le_modality = prepare_ml_features(train_df, localizers_df)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    
    # Train a simple model
    print("\nTraining machine learning model...")
    model, scaler, feature_importance = train_simple_model(X, y)
    
    return train_df, localizers_df, X, y, le_sex, le_modality, model, scaler

# Utility functions for working with coordinates
def extract_coordinates_info(localizers_df):
    """Extract detailed coordinate information"""
    
    coord_info = []
    for _, row in localizers_df.iterrows():
        try:
            coord_dict = ast.literal_eval(row['coordinates'])
            info = {
                'SeriesInstanceUID': row['SeriesInstanceUID'],
                'SOPInstanceUID': row['SOPInstanceUID'],
                'location': row['location'],
                'x': coord_dict.get('x'),
                'y': coord_dict.get('y'),
                'f': coord_dict.get('f')  # Some coordinates have a 'f' parameter
            }
            coord_info.append(info)
        except Exception as e:
            print(f"Error parsing coordinates: {row['coordinates']}, Error: {e}")
    
    return pd.DataFrame(coord_info)

# Summary statistics function
def generate_summary_report(train_df, localizers_df):
    """Generate a comprehensive summary report"""
    
    report = {
        'dataset_info': {
            'total_patients': len(train_df),
            'patients_with_aneurysms': train_df['Aneurysm Present'].sum(),
            'aneurysm_prevalence': train_df['Aneurysm Present'].mean(),
            'total_localizers': len(localizers_df),
            'unique_series_with_localizers': localizers_df['SeriesInstanceUID'].nunique()
        },
        'demographics': {
            'age_stats': train_df['PatientAge'].describe().to_dict(),
            'sex_distribution': train_df['PatientSex'].value_counts().to_dict(),
            'modality_distribution': train_df['Modality'].value_counts().to_dict()
        },
        'aneurysm_locations': {}
    }
    
    # Calculate location statistics
    aneurysm_columns = [col for col in train_df.columns if col not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
    for col in aneurysm_columns:
        report['aneurysm_locations'][col] = {
            'count': int(train_df[col].sum()),
            'percentage': float(train_df[col].mean() * 100)
        }
    
    return report

# Save results to CSV
def save_results_to_csv(train_df, localizers_df, feature_importance):
    """Save analysis results to CSV files"""
    
    # Save processed coordinate data
    coord_df = extract_coordinates_info(localizers_df)
    coord_df.to_csv('processed_coordinates.csv', index=False)
    print("Saved processed coordinates to 'processed_coordinates.csv'")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("Saved feature importance to 'feature_importance.csv'")
    
    # Save summary statistics
    summary = generate_summary_report(train_df, localizers_df)
    
    # Convert summary to DataFrame for CSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('summary_report.csv', index=False)
    print("Saved summary report to 'summary_report.csv'")

if __name__ == "__main__":
    try:
        # Run the complete analysis
        results = main_analysis()
        train_df, localizers_df, X, y, le_sex, le_modality, model, scaler = results
        
        # Generate summary report
        summary = generate_summary_report(train_df, localizers_df)
        print("\n=== SUMMARY REPORT ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Save results
        print("\nSaving results to CSV files...")
        # Get feature importance from the trained model
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        save_results_to_csv(train_df, localizers_df, feature_importance)
        
        print("\n✅ Analysis completed successfully!")
        print("Generated files:")
        print("- processed_coordinates.csv")
        print("- feature_importance.csv") 
        print("- summary_report.csv")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please check your CSV files are in the same directory as this script.")
