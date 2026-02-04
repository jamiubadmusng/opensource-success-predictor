# Analysis Report: Predicting Open-Source Project Success

## Using Early Activity Signals to Forecast GitHub Repository Growth

**Author:** Jamiu Olamilekan Badmus  
**Date:** February 2026

---

## 1. Introduction

The open-source software ecosystem has become a cornerstone of modern technology development. GitHub, the largest code hosting platform, hosts over 200 million repositories, with thousands of new projects created daily. For stakeholders across the technology landscape, identifying which projects will gain meaningful traction presents both an opportunity and a challenge.

This analysis develops a predictive model to identify promising open-source projects early in their lifecycle, specifically within the first 30 days after creation. The ability to forecast project success has practical applications for technology scouts evaluating emerging tools, venture capitalists assessing developer tool investments, and open-source program offices allocating sponsorship resources.

### 1.1 Research Objectives

1. Determine whether early activity signals (first 30 days) can reliably predict project success at the 6-month mark
2. Identify the most predictive features for project success
3. Develop a deployable machine learning model for scoring new repositories
4. Generate actionable insights for technology decision-makers

### 1.2 Success Definition

A project is classified as "successful" if it meets either of the following criteria within six months:

- **Popularity criterion**: Accumulates 50 or more stars, indicating significant community interest
- **Sustainability criterion**: Attracts 3 or more unique contributors AND maintains development activity in months 4-6

This dual definition captures both viral attention (stars) and organic community growth (contributors with sustained activity).

---

## 2. Data Overview

### 2.1 Data Source

The dataset was extracted from GH Archive, a public record of all GitHub activity since 2011. The archive captures every public event including pushes, pull requests, issues, comments, stars, and forks. Data was queried from Google BigQuery's public dataset, which provides efficient access to the 17+ terabyte archive.

### 2.2 Dataset Characteristics

The analysis covers repositories created in Q1 2024 (January through March), with outcome measurements taken through Q3 2024 (September). This provides a full six-month observation window for all repositories.

| Attribute | Value |
|-----------|-------|
| Total repositories | 34,664 |
| Time period (creation) | January - March 2024 |
| Time period (outcomes) | Through September 2024 |
| Feature categories | Early activity, metadata, outcomes |
| Class distribution | 43.52% successful, 56.48% not successful (ratio 1.30:1) |

### 2.3 Feature Categories

**Early Activity Features (First 30 Days):**
- Commit and push activity
- Issue events (opened, closed, commented)
- Pull request events (opened, closed, reviewed)
- Community engagement (stars, forks)
- Contributor metrics
- Weekly activity distribution

**Metadata Features:**
- Primary programming language
- Presence of repository description
- Creator information

**Outcome Features (6 Months):**
- Cumulative stars and forks
- Total contributors
- Sustained activity in months 4-6
- Release count

---

## 3. Exploratory Analysis

### 3.1 Target Variable Distribution

The dataset exhibits a relatively balanced class distribution, with 43.52% of projects classified as "successful" and 56.48% as "not successful" (class ratio of 1.30:1). This better-than-expected balance is a positive finding for model development.

Key implications for modeling:
- Standard accuracy metrics are valid but should be supplemented with precision/recall
- SMOTE resampling is not necessary given the balanced class distribution
- Equal focus can be given to both classes during model training

### 3.2 Programming Language Analysis

Programming language distribution reveals Python, JavaScript, and TypeScript as the most common languages among new repositories. However, success rates vary considerably by language:

Languages with higher success rates tend to be those associated with current technology trends (Rust, Go, TypeScript), while more established languages show more moderate success rates. This pattern suggests that projects in emerging technology areas may have a baseline advantage in attracting attention.

### 3.3 Early Activity Patterns

Comparison of early activity between successful and unsuccessful projects reveals several distinguishing patterns:

**Stars and Forks:** Successful projects accumulate stars and forks at significantly higher rates in their first 30 days. The median star count for successful projects is typically 5-10x higher than unsuccessful projects.

**Contributor Activity:** Successful projects attract more unique contributors early on. The presence of multiple contributors in the first month is a strong positive indicator.

**Issue and PR Activity:** Active issue trackers and pull request activity indicate both external interest and healthy project governance. Successful projects show higher rates of issue commenting and PR reviews.

**Activity Trends:** The distribution of activity across the four weeks provides insights into project momentum. Successful projects tend to maintain or increase activity over time, while unsuccessful projects often show declining activity after an initial burst.

### 3.4 Feature Correlations

Correlation analysis reveals several important patterns:

**High correlations with success:**
- Stars in first 30 days (r = 0.45-0.55)
- Unique contributors (r = 0.35-0.45)
- Fork count (r = 0.30-0.40)
- Total events (r = 0.25-0.35)

**Feature interdependencies:**
- Commits and pushes are highly correlated (expected)
- Stars and forks show moderate correlation (community interest)
- Issue and PR activity are moderately correlated (project maturity)

These correlations inform feature engineering decisions and help identify potential multicollinearity issues for certain model types.

---

## 4. Feature Engineering

### 4.1 Engineered Features

Beyond the raw activity counts, several derived features were created to capture meaningful patterns:

**Activity Ratios:**
- Commits per push event (development style indicator)
- Events per contributor (activity density)
- Issue close rate (project responsiveness)
- PR close rate (contribution handling)

**Community Indicators:**
- Has external interest (stars or forks present)
- Has multiple contributors
- Has issue activity
- Has PR activity

**Trend Features:**
- Week-over-week activity ratios
- Activity sustainability (later weeks vs. earlier weeks)

**Engagement Depth:**
- Comments per issue (discussion intensity)
- Star-to-fork ratio (viral vs. utilitarian interest)

### 4.2 Feature Selection Rationale

The engineered features aim to capture aspects of project health not evident from raw counts:

1. **Ratio features** normalize for project size and reveal efficiency patterns
2. **Binary indicators** identify presence of community engagement types
3. **Trend features** capture momentum and sustainability signals
4. **Depth features** measure quality of engagement beyond quantity

---

## 5. Model Development

### 5.1 Data Preprocessing

Given the relatively balanced class distribution (1.30:1 ratio), no resampling techniques were required. The natural data distribution was preserved for training, which helps maintain realistic decision boundaries.

Key preprocessing steps:
- Feature scaling using StandardScaler for non-binary features
- Train-test split (80-20) with stratification
- Feature engineering to create 35 predictive features from raw activity data

### 5.2 Models Evaluated

Six classification algorithms were evaluated:

1. **Logistic Regression**: Linear model serving as interpretable baseline
2. **Decision Tree**: Single tree for interpretability benchmark
3. **Random Forest**: Ensemble of decision trees with bagging
4. **Gradient Boosting**: Sequential ensemble with boosting
5. **XGBoost**: Optimized gradient boosting implementation
6. **LightGBM**: Efficient gradient boosting with leaf-wise growth

### 5.3 Evaluation Framework

Models were evaluated using:

- **5-fold stratified cross-validation** on training data
- **Held-out test set** (20% of data) for final evaluation
- **ROC-AUC** as primary ranking metric
- **Precision, Recall, F1** for operational assessment

The stratified approach ensures consistent class distributions across all folds and the test set.

---

## 6. Results

### 6.1 Model Performance Summary

| Model | CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test F1 |
|-------|------------|--------------|---------------|---------|
| Gradient Boosting | 0.9213 | **0.9131** | 0.8165 | 0.7830 |
| LightGBM | 0.9209 | 0.9122 | 0.8093 | 0.7713 |
| XGBoost | 0.9195 | 0.9108 | 0.8111 | 0.7747 |
| Random Forest | 0.9120 | 0.9029 | 0.8058 | 0.7668 |
| Logistic Regression | 0.8951 | 0.8881 | 0.7967 | 0.7505 |
| Decision Tree | 0.8075 | 0.8099 | 0.7672 | 0.7194 |

Key observations:
- **Gradient Boosting** achieved the best test ROC-AUC of **0.9131**
- Ensemble methods significantly outperform single models (Decision Tree)
- Boosting methods slightly outperform bagging (Random Forest)
- Logistic Regression provides reasonable performance (ROC-AUC 0.8881) with full interpretability

### 6.2 Classification Performance

For the best-performing model (Gradient Boosting) at the default 0.5 threshold:

| Metric | Not Successful | Successful | Overall |
|--------|---------------|------------|---------|
| Precision | 0.82 | 0.81 | 0.82 |
| Recall | 0.86 | 0.76 | 0.82 |
| F1-Score | 0.84 | 0.78 | 0.81 |

The confusion matrix analysis reveals:
- **True Negatives (TN)**: 3,366 - Correctly identified unsuccessful projects
- **False Positives (FP)**: 550 - Incorrectly flagged as successful
- **False Negatives (FN)**: 722 - Missed successful projects
- **True Positives (TP)**: 2,295 - Correctly identified successful projects
- **Overall Accuracy**: 81.65%

### 6.3 Threshold Analysis

Different probability thresholds enable different operational trade-offs:

| Threshold | Projects Flagged | % of Test Set | Precision | Recall |
|-----------|------------------|---------------|-----------|--------|
| 0.3 | 3,936 | 56.77% | 0.6944 | 0.9059 |
| 0.4 | 3,382 | 48.78% | 0.7501 | 0.8409 |
| 0.5 | 2,845 | 41.04% | 0.8067 | 0.7607 |
| 0.6 | 2,385 | 34.40% | 0.8604 | 0.6801 |
| 0.7 | 2,010 | 28.99% | 0.9015 | 0.6006 |
| 0.8 | 1,573 | 22.69% | 0.9612 | 0.5012 |

For technology scouting (where false positives are costly), higher thresholds (0.6-0.7) are recommended. For broad monitoring (where missing projects is costly), lower thresholds (0.3-0.4) may be appropriate.

---

## 7. Feature Importance Analysis

### 7.1 Top Predictive Features

Both model-based feature importance and SHAP analysis identify consistent top predictors:

| Rank | Feature | Importance | Mean (Successful) | Mean (Not Successful) |
|------|---------|------------|-------------------|----------------------|
| 1 | **total_events_30d** | 0.5028 | 171.68 | 61.97 |
| 2 | **stars_30d** | 0.1602 | 89.78 | 14.38 |
| 3 | **total_commits_30d** | 0.0732 | 107.30 | 41.67 |
| 4 | **activity_sustainability** | 0.0523 | 1.24 | 0.37 |
| 5 | **committer_ratio** | 0.0415 | 0.18 | 0.12 |
| 6 | **events_week_4** | 0.0401 | 23.80 | 4.38 |
| 7 | **events_per_contributor** | 0.0278 | 11.20 | 7.35 |
| 8 | **star_fork_ratio** | 0.0127 | 62.24 | 12.72 |

**Key Findings:**
1. **Total events in first 30 days**: The strongest single predictor (50.28% importance). This aggregate measure captures overall project activity volume.

2. **Stars in first 30 days**: Second most important (16.02%). Successful projects average 89.78 stars vs. 14.38 for unsuccessful projects.

3. **Total commits**: Development intensity matters (7.32% importance). Successful projects show 2.5x more commits.

4. **Activity sustainability**: Projects maintaining activity across all four weeks show healthier trajectories. Successful projects have 3.4x higher sustainability scores.

5. **Committer ratio**: Diversity of contributors relative to commit volume indicates healthy collaboration patterns.

### 7.2 SHAP Analysis Insights

SHAP (SHapley Additive exPlanations) values provide instance-level feature impact, offering deeper interpretability than simple feature importance:

**Top SHAP Features by Mean Absolute Value:**
1. **stars_30d** - Mean |SHAP| ≈ 1.30 (highest individual impact)
2. **total_events_30d** - Mean |SHAP| ≈ 1.05
3. **total_commits_30d** - Mean |SHAP| ≈ 0.22
4. **activity_sustainability** - Mean |SHAP| ≈ 0.20
5. **events_week_4** - Mean |SHAP| ≈ 0.15

**Key Interpretations:**
- **High star counts** have strong positive impact across all predictions (visible as pink/red points on positive SHAP side)
- **Low event counts** consistently push predictions toward "not successful" (blue points on negative side)
- **Sustained activity** (activity_sustainability > 1) strongly predicts success
- **Week 4 activity** is particularly diagnostic - projects with declining activity by week 4 rarely succeed

### 7.3 Feature Interaction Effects

Notable interaction effects include:
- Stars combined with multiple contributors has amplified positive effect
- High activity without external interest (stars/forks) has limited positive impact
- Issue activity without closure/response has negative association

---

## 8. Business Implications

### 8.1 For Technology Scouts

**High-confidence signals for promising projects:**
- 10+ stars in first two weeks
- 3+ unique contributors
- Active issue tracker with responses
- Consistent weekly activity (not declining)

**Red flags suggesting limited potential:**
- Zero external engagement (stars, forks, issues)
- Single contributor with declining activity
- No documentation (description, README)

### 8.2 For Open-Source Program Offices

**Scoring framework for sponsorship decisions:**
- Use 0.6-0.7 probability threshold for high-confidence candidates
- Prioritize projects with both stars AND contributor diversity
- Consider programming language trends in evaluation

**Resource allocation:**
- Focus deeper evaluation on top-scoring projects
- Use model scores to prioritize from large candidate lists

### 8.3 For Developers

**Indicators for contribution decisions:**
- Sustained activity suggests ongoing maintenance
- Multiple contributors indicates collaborative environment
- Active issue discussions show responsive maintainers

---

## 9. Limitations and Considerations

### 9.1 Data Limitations

- **Observation period**: Results based on Q1 2024 creation cohort may not generalize to all time periods
- **Public repositories only**: Private repositories and enterprise patterns not captured
- **Survivorship considerations**: Deleted repositories not included in analysis

### 9.2 Model Limitations

- **Binary classification**: Success is spectrum, not binary; threshold effects may miss nuance
- **Feature availability**: Some potentially predictive data (README quality, test coverage) not included
- **Temporal dynamics**: Static 30-day snapshot may miss important trajectory patterns

### 9.3 Ethical Considerations

- **Self-fulfilling predictions**: Publishing predictions could influence outcomes
- **Bias propagation**: Model may amplify existing biases in open-source visibility
- **Definitional choices**: Success definition favors certain project types

---

## 10. Conclusions

This analysis demonstrates that early activity signals provide meaningful predictive power for identifying open-source projects likely to succeed. The best-performing Gradient Boosting model achieves a **ROC-AUC of 0.9131** and **accuracy of 81.65%**, enabling practical applications for technology decision-making.

### 10.1 Key Takeaways

1. **Early signals are highly predictive**: First 30 days of activity contain substantial information about future trajectories (ROC-AUC > 0.91)
2. **Total activity volume matters most**: `total_events_30d` is the strongest predictor (50.28% importance), followed by star count (16.02%)
3. **Sustainability over volume**: Activity sustainability ratio differentiates successful projects (mean 1.24) from unsuccessful (mean 0.37)
4. **Gradient boosting excels**: Gradient Boosting achieved best performance, with LightGBM and XGBoost close behind
5. **Balanced classes**: The dataset showed better class balance (43.52% successful) than expected, simplifying model development

### 10.2 Recommendations

For practical deployment:
1. Use the trained model to score repositories weekly
2. Set appropriate thresholds based on use case (high precision vs. high recall)
3. Combine model scores with qualitative assessment for final decisions
4. Monitor model performance over time and retrain as ecosystem evolves

### 10.3 Future Directions

Promising extensions include:
1. Incorporating text features from README and descriptions
2. Adding creator reputation metrics
3. Developing time-series models for trajectory prediction
4. Building real-time scoring infrastructure

---

## References

1. GH Archive. (2024). GitHub Archive. https://www.gharchive.org/
2. Hoffa, F. (2023). Querying GitHub Archive with Snowflake: The Essentials. Medium.
3. Borges, H., Hora, A., & Valente, M. T. (2016). Understanding the Factors that Impact the Popularity of GitHub Repositories. ICSME 2016.
4. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS 2017.

---

*This analysis was conducted as part of a data science portfolio project. For questions or collaboration opportunities, contact jamiubadmus001@gmail.com.*
