"""
Predicting Open-Source Project Success
=======================================

This module provides functions to predict whether a newly created GitHub repository
will become successful within six months, using only activity data from the first 
30 days after creation.

Author: Jamiu Olamilekan Badmus
Email: jamiubadmus001@gmail.com
GitHub: https://github.com/jamiubadmusng

Usage:
------
    python predict_success.py --input data/raw/github_projects_dataset.csv

"""

import pandas as pd
import numpy as np
import warnings
import argparse
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the GitHub projects dataset from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the dataset.
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]:,} records with {df.shape[1]} columns.")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw dataset for modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
        
    Returns
    -------
    pd.DataFrame
        Preprocessed dataset with encoded features.
    """
    df_processed = df.copy()
    
    # Handle missing values
    df_processed['primary_language'] = df_processed['primary_language'].fillna('Unknown')
    df_processed['has_description'] = df_processed['description'].notna().astype(int)
    
    # Group rare languages
    language_threshold = 100
    language_counts = df_processed['primary_language'].value_counts()
    rare_languages = language_counts[language_counts < language_threshold].index.tolist()
    
    df_processed['language_grouped'] = df_processed['primary_language'].apply(
        lambda x: 'Other' if x in rare_languages else x
    )
    
    # One-hot encode languages
    language_dummies = pd.get_dummies(df_processed['language_grouped'], prefix='lang')
    df_processed = pd.concat([df_processed, language_dummies], axis=1)
    
    return df_processed


# =============================================================================
# Feature Engineering
# =============================================================================

def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from early activity metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the original features.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """
    df_eng = df.copy()
    
    # Activity intensity ratios
    df_eng['commits_per_push'] = np.where(
        df_eng['push_events_30d'] > 0,
        df_eng['total_commits_30d'] / df_eng['push_events_30d'],
        0
    )
    
    df_eng['events_per_contributor'] = np.where(
        df_eng['unique_contributors_30d'] > 0,
        df_eng['total_events_30d'] / df_eng['unique_contributors_30d'],
        0
    )
    
    # Issue resolution efficiency
    df_eng['issue_close_rate'] = np.where(
        df_eng['issues_opened_30d'] > 0,
        df_eng['issues_closed_30d'] / df_eng['issues_opened_30d'],
        0
    )
    
    # PR merge efficiency
    df_eng['pr_close_rate'] = np.where(
        df_eng['prs_opened_30d'] > 0,
        df_eng['prs_closed_30d'] / df_eng['prs_opened_30d'],
        0
    )
    
    # Community engagement indicators
    df_eng['has_external_interest'] = (
        (df_eng['stars_30d'] > 0) | (df_eng['forks_30d'] > 0)
    ).astype(int)
    df_eng['has_issues'] = (df_eng['issues_opened_30d'] > 0).astype(int)
    df_eng['has_prs'] = (df_eng['prs_opened_30d'] > 0).astype(int)
    df_eng['has_multiple_contributors'] = (df_eng['unique_contributors_30d'] > 1).astype(int)
    
    # Activity trend features
    df_eng['week1_to_week2_ratio'] = np.where(
        df_eng['events_week_1'] > 0,
        df_eng['events_week_2'] / df_eng['events_week_1'],
        0
    )
    
    df_eng['week3_to_week4_ratio'] = np.where(
        df_eng['events_week_3'] > 0,
        df_eng['events_week_4'] / df_eng['events_week_3'],
        0
    )
    
    # Activity sustainability
    early_weeks = df_eng['events_week_1'] + df_eng['events_week_2']
    later_weeks = df_eng['events_week_3'] + df_eng['events_week_4']
    df_eng['activity_sustainability'] = np.where(
        early_weeks > 0,
        later_weeks / early_weeks,
        0
    )
    
    # Committer diversity
    df_eng['committer_ratio'] = np.where(
        df_eng['unique_contributors_30d'] > 0,
        df_eng['unique_committers_30d'] / df_eng['unique_contributors_30d'],
        0
    )
    
    # Engagement depth
    df_eng['comments_per_issue'] = np.where(
        df_eng['issue_events_30d'] > 0,
        df_eng['issue_comments_30d'] / df_eng['issue_events_30d'],
        0
    )
    
    # Star to fork ratio
    df_eng['star_fork_ratio'] = np.where(
        df_eng['forks_30d'] > 0,
        df_eng['stars_30d'] / df_eng['forks_30d'],
        df_eng['stars_30d']
    )
    
    # Handle infinite values
    df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
    df_eng = df_eng.fillna(0)
    
    return df_eng


def get_feature_lists(df_processed: pd.DataFrame) -> tuple:
    """
    Get lists of features for modeling.
    
    Parameters
    ----------
    df_processed : pd.DataFrame
        Preprocessed dataset.
        
    Returns
    -------
    tuple
        Tuple containing (early_activity_features, metadata_features, 
        language_features, engineered_features, all_features).
    """
    early_activity_features = [
        'push_events_30d', 'total_commits_30d', 'issue_events_30d',
        'issues_opened_30d', 'issues_closed_30d', 'issue_comments_30d',
        'pr_events_30d', 'prs_opened_30d', 'prs_closed_30d', 'pr_reviews_30d',
        'stars_30d', 'forks_30d', 'unique_contributors_30d', 'unique_committers_30d',
        'events_week_1', 'events_week_2', 'events_week_3', 'events_week_4',
        'total_events_30d'
    ]
    
    metadata_features = ['has_description']
    
    language_features = [col for col in df_processed.columns if col.startswith('lang_')]
    
    engineered_features = [
        'commits_per_push', 'events_per_contributor', 'issue_close_rate', 
        'pr_close_rate', 'has_external_interest', 'has_issues', 'has_prs', 
        'has_multiple_contributors', 'week1_to_week2_ratio', 'week3_to_week4_ratio', 
        'activity_sustainability', 'committer_ratio', 'comments_per_issue', 
        'star_fork_ratio'
    ]
    
    all_features = (
        early_activity_features + 
        metadata_features + 
        language_features + 
        engineered_features
    )
    
    return (
        early_activity_features, 
        metadata_features, 
        language_features, 
        engineered_features, 
        all_features
    )


# =============================================================================
# Model Training and Evaluation
# =============================================================================

def train_models(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> dict:
    """
    Train multiple models and return evaluation results.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
        
    Returns
    -------
    dict
        Dictionary containing trained models and their results.
    """
    # Handle class imbalance
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Training set after SMOTE: {len(y_train_resampled):,} samples")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42,
            class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric='logloss', use_label_encoder=False
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            class_weight='balanced', verbose=-1
        )
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_resampled, y_train_resampled,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        # Train on full training set
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'model': model,
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std(),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test ROC-AUC: {results[name]['test_roc_auc']:.4f}")
    
    return results


def select_best_model(results: dict) -> tuple:
    """
    Select the best model based on test ROC-AUC.
    
    Parameters
    ----------
    results : dict
        Dictionary containing model results.
        
    Returns
    -------
    tuple
        Tuple containing (best_model_name, best_model, best_results).
    """
    best_name = max(results, key=lambda x: results[x]['test_roc_auc'])
    best_model = results[best_name]['model']
    best_results = results[best_name]
    
    print(f"\nBest Model: {best_name}")
    print(f"  Test ROC-AUC: {best_results['test_roc_auc']:.4f}")
    print(f"  Test F1: {best_results['test_f1']:.4f}")
    
    return best_name, best_model, best_results


def print_classification_report(y_test: pd.Series, y_pred: np.ndarray, model_name: str):
    """
    Print detailed classification report.
    
    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    model_name : str
        Name of the model.
    """
    print(f"\nClassification Report for {model_name}:")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred, 
        target_names=['Not Successful', 'Successful']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix Breakdown:")
    print(f"  True Negatives: {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives: {tp:,}")


# =============================================================================
# Model Persistence
# =============================================================================

def save_model(model, scaler, model_dir: str = 'models'):
    """
    Save trained model and scaler to disk.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    scaler : StandardScaler
        Fitted scaler.
    model_dir : str, optional
        Directory to save models, by default 'models'.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'best_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    print(f"\nModel saved to {model_dir}/best_model.joblib")
    print(f"Scaler saved to {model_dir}/scaler.joblib")


def load_model(model_dir: str = 'models') -> tuple:
    """
    Load trained model and scaler from disk.
    
    Parameters
    ----------
    model_dir : str, optional
        Directory containing saved models, by default 'models'.
        
    Returns
    -------
    tuple
        Tuple containing (model, scaler).
    """
    model = joblib.load(os.path.join(model_dir, 'best_model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    
    return model, scaler


# =============================================================================
# Main Pipeline
# =============================================================================

def main(input_path: str, output_dir: str = None):
    """
    Run the complete prediction pipeline.
    
    Parameters
    ----------
    input_path : str
        Path to input CSV file.
    output_dir : str, optional
        Directory for outputs, by default None.
    """
    print("=" * 70)
    print("PREDICTING OPEN-SOURCE PROJECT SUCCESS")
    print("=" * 70)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = load_data(input_path)
    
    # Preprocess data
    print("\n[2/6] Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Get feature lists
    (
        early_activity_features, 
        metadata_features, 
        language_features, 
        engineered_features, 
        all_features
    ) = get_feature_lists(df_processed)
    
    # Create initial feature matrix
    initial_features = early_activity_features + metadata_features + language_features
    X = df_processed[initial_features].copy()
    y = df_processed['is_successful_project'].copy()
    
    # Train-test split
    print("\n[3/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # Feature engineering
    print("\n[4/6] Engineering features...")
    X_train_eng = create_engineered_features(X_train)
    X_test_eng = create_engineered_features(X_test)
    
    # Scale features
    binary_features = (
        ['has_description', 'has_external_interest', 'has_issues', 
         'has_prs', 'has_multiple_contributors'] + language_features
    )
    features_to_scale = [col for col in X_train_eng.columns if col not in binary_features]
    
    scaler = StandardScaler()
    X_train_scaled = X_train_eng.copy()
    X_test_scaled = X_test_eng.copy()
    
    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train_eng[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test_eng[features_to_scale])
    
    # Train models
    print("\n[5/6] Training models...")
    results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Select best model
    best_name, best_model, best_results = select_best_model(results)
    
    # Print detailed report
    print_classification_report(y_test, best_results['y_pred'], best_name)
    
    # Save model
    print("\n[6/6] Saving model...")
    model_dir = os.path.join(os.path.dirname(input_path), '..', 'models')
    save_model(best_model, scaler, model_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nBest Model: {best_name}")
    print(f"Test ROC-AUC: {best_results['test_roc_auc']:.4f}")
    print(f"Test F1 Score: {best_results['test_f1']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict open-source project success from early activity metrics.'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Directory for output files'
    )
    
    args = parser.parse_args()
    main(args.input, args.output)
