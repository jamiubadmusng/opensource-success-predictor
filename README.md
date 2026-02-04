# Predicting Open-Source Project Success

## Using Early Activity Signals to Forecast GitHub Repository Growth

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data Source](https://img.shields.io/badge/Data-GH%20Archive-orange.svg)](https://www.gharchive.org/)

**Author:** Jamiu Olamilekan Badmus  
**Email:** jamiubadmus001@gmail.com  
**LinkedIn:** [Jamiu Olamilekan Badmus](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)  
**GitHub:** [jamiubadmusng](https://github.com/jamiubadmusng)

---

## Executive Summary

This project develops a machine learning model to predict which newly created GitHub repositories will become successful within six months, using only activity data from their first 30 days. The model achieves strong predictive performance and provides actionable insights for technology scouts, open-source program offices, and investors seeking to identify promising open-source projects early in their lifecycle.

**Key Results:**
- Trained and evaluated 6 classification models on 34,000+ GitHub repositories
- Identified early signals most predictive of project success (stars, contributors, issue activity)
- Developed a deployable model for scoring new repositories in real-time

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Data Source](#data-source)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Key Findings](#key-findings)
9. [Future Work](#future-work)
10. [References](#references)

---

## Problem Statement

The open-source software ecosystem hosts millions of repositories, with thousands of new projects created daily. Identifying which projects will gain traction is valuable for:

- **Technology scouts** evaluating emerging tools and frameworks
- **Open-source program offices** allocating sponsorship resources
- **Venture capitalists** identifying promising developer tools
- **Developers** deciding which projects to contribute to

This project addresses the question: **Can we predict which open-source projects will become successful within six months using only their first 30 days of activity data?**

### Success Definition

A repository is classified as "successful" if it meets any of the following criteria by the six-month mark:

1. Accumulates 50 or more stars (indicating community interest)
2. Attracts 3 or more unique contributors AND shows continued development activity in months 4-6 (indicating sustainable growth)

---

## Data Source

The dataset was extracted from [GH Archive](https://www.gharchive.org/), a comprehensive record of public GitHub activity since 2011. GH Archive captures every public event on GitHub, including:

- Push events (commits)
- Pull requests (opened, closed, merged)
- Issues (opened, closed, commented)
- Watch events (stars)
- Fork events

### Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| Source | GH Archive via Google BigQuery |
| Time Period | Q1 2024 (repository creation) to Q3 2024 (outcome measurement) |
| Total Records | 34,664 repositories |
| Features | 40 columns (19 early activity, 10 outcomes, 11 metadata) |
| Target Variable | Binary classification (Successful / Not Successful) |

### Data Extraction

SQL queries for BigQuery are provided in the `queries/` directory. The main query (`05_combined_dataset.sql`) extracts:

- Repository metadata (language, description, creator)
- Early activity metrics (first 30 days)
- Outcome metrics (6-month performance)
- Pre-computed success labels

---

## Project Structure

```
tech/
├── data/
│   ├── raw/                          # Original dataset from BigQuery
│   │   └── github_projects_dataset.csv
│   └── processed/                    # Cleaned and feature-engineered data
│       └── github_projects_processed.csv
├── docs/
│   ├── analysis_report.md            # Detailed analysis write-up
│   └── figures/                      # Visualization outputs
│       ├── target_distribution.png
│       ├── correlation_heatmap.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_pr_curves.png
│       ├── feature_importance.png
│       └── shap_beeswarm.png
├── models/                           # Trained model artifacts
│   ├── best_model.joblib
│   └── scaler.joblib
├── notebooks/
│   └── predicting_opensource_success.ipynb  # Main analysis notebook
├── queries/                          # BigQuery SQL scripts
│   ├── 01_extract_new_repositories.sql
│   ├── 02_early_activity_metrics.sql
│   ├── 03_outcome_metrics.sql
│   ├── 04_cumulative_stars.sql
│   ├── 05_combined_dataset.sql       # Main query
│   └── README_BIGQUERY_INSTRUCTIONS.md
├── src/
│   └── predict_success.py            # Standalone Python module
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Google Cloud account for BigQuery access

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jamiubadmusng/portfolio.git
   cd portfolio/tech
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
xgboost>=1.7.0
lightgbm>=3.3.0
shap>=0.41.0
imbalanced-learn>=0.10.0
jupyter>=1.0.0
joblib>=1.2.0
```

---

## Usage

### Running the Jupyter Notebook

The main analysis is contained in the Jupyter notebook:

```bash
cd notebooks
jupyter notebook predicting_opensource_success.ipynb
```

### Running the Python Script

For command-line execution:

```bash
python src/predict_success.py --input data/raw/github_projects_dataset.csv
```

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/best_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Prepare new data (must have same features as training data)
new_data = pd.read_csv('new_repositories.csv')
# ... preprocess and engineer features ...

# Make predictions
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)[:, 1]
```

---

## Methodology

### 1. Data Preprocessing

- Handled missing values in `primary_language` (replaced with 'Unknown')
- Created binary indicator for projects with descriptions
- Grouped rare programming languages into 'Other' category
- One-hot encoded programming languages

### 2. Feature Engineering

Created 14 additional features from early activity metrics:

| Feature | Description |
|---------|-------------|
| `commits_per_push` | Average commits per push event |
| `events_per_contributor` | Activity density per contributor |
| `issue_close_rate` | Ratio of closed to opened issues |
| `pr_close_rate` | Ratio of closed to opened PRs |
| `has_external_interest` | Binary: has stars or forks |
| `has_issues` | Binary: has opened issues |
| `has_prs` | Binary: has opened PRs |
| `has_multiple_contributors` | Binary: more than one contributor |
| `week1_to_week2_ratio` | Activity trend (week 2 / week 1) |
| `week3_to_week4_ratio` | Activity trend (week 4 / week 3) |
| `activity_sustainability` | Later weeks / earlier weeks ratio |
| `committer_ratio` | Committers / total contributors |
| `comments_per_issue` | Engagement depth on issues |
| `star_fork_ratio` | Viral potential indicator |

### 3. Class Balance Analysis

The dataset showed relatively balanced classes (43.52% successful vs 56.48% not successful, ratio 1.30:1), eliminating the need for resampling techniques like SMOTE. The natural class distribution was preserved for training.

### 4. Model Training

Evaluated six classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. LightGBM

Used 5-fold stratified cross-validation for model selection.

### 5. Evaluation Metrics

- **ROC-AUC**: Primary metric for model ranking
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1 Score**: Harmonic mean of precision and recall

---

## Results

### Model Performance Comparison

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 | Test Precision | Test Recall |
|-------|------------|--------------|---------|----------------|-------------|
| **Gradient Boosting** | **0.9213** | **0.9131** | **0.7830** | **0.8067** | **0.7607** |
| LightGBM | 0.9209 | 0.9122 | 0.7713 | 0.8034 | 0.7414 |
| XGBoost | 0.9195 | 0.9108 | 0.7747 | 0.7990 | 0.7518 |
| Random Forest | 0.9120 | 0.9029 | 0.7668 | 0.7954 | 0.7401 |
| Logistic Regression | 0.8951 | 0.8881 | 0.7505 | 0.7727 | 0.7294 |
| Decision Tree | 0.8075 | 0.8099 | 0.7194 | 0.7367 | 0.7029 |

**Best Model**: Gradient Boosting with ROC-AUC of 0.9131 (91.31% probability of ranking a successful project higher than an unsuccessful one)

### Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `total_events_30d` | 0.5028 |
| 2 | `stars_30d` | 0.1602 |
| 3 | `total_commits_30d` | 0.0732 |
| 4 | `activity_sustainability` | 0.0523 |
| 5 | `committer_ratio` | 0.0415 |
| 6 | `events_week_4` | 0.0401 |
| 7 | `events_per_contributor` | 0.0278 |
| 8 | `week3_to_week4_ratio` | 0.0150 |
| 9 | `star_fork_ratio` | 0.0127 |
| 10 | `week1_to_week2_ratio` | 0.0120 |

---

## Key Findings

### 1. Total Activity Volume is the Strongest Predictor

The `total_events_30d` feature accounts for 50.28% of model importance. Projects with high overall activity (commits, issues, PRs, stars) in their first month are significantly more likely to succeed. Successful projects average 171.7 events vs 62.0 for unsuccessful projects.

### 2. Early Stars are Highly Predictive

Stars in the first 30 days (`stars_30d`) is the second most important feature at 16.02%. Successful projects average 89.8 stars vs 14.4 for unsuccessful ones—a 6.2x difference.

### 3. Activity Sustainability Over Volume

The `activity_sustainability` ratio (weeks 3-4 activity divided by weeks 1-2 activity) is the 4th most important feature. Projects that maintain or grow their activity are more likely to succeed. Successful projects have a sustainability ratio of 1.24 vs 0.37 for unsuccessful projects.

### 4. Week 4 Activity is Diagnostic

Projects with strong Week 4 activity (`events_week_4`) are much more likely to succeed. This late-month activity indicates sustained momentum rather than a one-time burst.

### 5. Programming Language Effects

Success rates vary by programming language, with certain languages (Rust, TypeScript, Go) showing higher baseline success rates, potentially reflecting their positions in current technology trends.

---

## Future Work

1. **Text Analysis**: Incorporate NLP features from README files and repository descriptions
2. **Creator Features**: Add developer reputation metrics (prior successful projects, followers)
3. **Time Series Models**: Develop models that capture growth trajectory patterns
4. **Real-time Scoring API**: Build a web service for continuous monitoring of new projects
5. **Causal Analysis**: Investigate which early actions causally influence success

---

## References

1. GH Archive. (2024). *GitHub Archive*. Retrieved from https://www.gharchive.org/
2. Hoffa, F. (2023). *Querying GitHub Archive with Snowflake: The Essentials*. Medium.
3. Borges, H., et al. (2016). "Understanding the Factors that Impact the Popularity of GitHub Repositories." *ICSME 2016*.
4. Cosentino, V., et al. (2017). "A Systematic Mapping Study of Software Development With GitHub." *IEEE Access*.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- GH Archive for providing the public GitHub event data
- Google BigQuery for enabling efficient querying of large-scale data
- The open-source community for the tools and libraries used in this project

---

## Contact

For questions or collaboration opportunities:

- **Email**: jamiubadmus001@gmail.com
- **LinkedIn**: [Jamiu Olamilekan Badmus](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)
- **GitHub**: [jamiubadmusng](https://github.com/jamiubadmusng)
- **Website**: [https://sites.google.com/view/jamiu-olamilekan-badmus/](https://sites.google.com/view/jamiu-olamilekan-badmus/)
