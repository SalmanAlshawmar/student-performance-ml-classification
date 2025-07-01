# Machine Learning Project: Student Performance Classification

This project implements multiple machine learning algorithms to predict student performance using decision trees, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).

## Dataset

The project uses the **Student Performance Dataset** from the UCI Machine Learning Repository:
- **Dataset Link**: https://archive.ics.uci.edu/dataset/320/student+performance
- **Description**: This dataset contains student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features.

## Project Overview

This machine learning project analyzes student performance data to predict whether a student will pass or fail based on various features. The project implements and compares three different classification algorithms:

1. **Decision Tree Classifier**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**

## Features

- **Data Preprocessing**: One-hot encoding of categorical variables, feature scaling
- **Model Training**: Implementation of three different ML algorithms
- **Hyperparameter Tuning**: GridSearchCV for optimal parameter selection
- **Model Evaluation**: Confusion matrices, classification reports, accuracy scores
- **Visualization**: Decision tree plots, correlation heatmaps, hyperparameter analysis
- **Cross-Validation**: K-fold cross-validation for robust model evaluation

## Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

## Project Structure

```
ML_Pro/
├── ML Projectt.ipynb          # Main Jupyter notebook
├── student-mat.xls            # Dataset file
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── .gitignore                # Git ignore file
```

## Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd ML_Pro
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `ML Projectt.ipynb` and run the cells

## Model Performance

The project includes comprehensive model evaluation with:
- Training/Validation/Test splits (60%/20%/20%)
- Hyperparameter tuning using GridSearchCV
- Confusion matrices for each model
- Feature importance analysis
- Cross-validation scores

## Model Performance Results

| Model | Test Accuracy | Best Parameters |
|-------|---------------|-----------------|
| **Decision Tree** | 94.94% | criterion='gini', max_depth=5, min_samples_leaf=20 |
| **Support Vector Machine** | 92.41% | C=0.1, gamma='scale', kernel='linear' |
| **K-Nearest Neighbors** | 81.01% | n_neighbors=21, p=1, weights='distance' |

### Performance Summary

- **Best Overall Model**: Decision Tree (Tuned) with **94.94% test accuracy**
- **Cross-Validation Best**: Decision Tree with **92.81% CV score**
- **Most Consistent**: SVM (Tuned) with good balance between complexity and performance

### Key Results

- **Decision Tree**: Includes feature importance analysis and tree visualization
- **KNN**: Optimal k-value selection through grid search  
- **SVM**: Linear and RBF kernel comparison with 2D PCA visualization
