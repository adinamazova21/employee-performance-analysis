import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def load_data():
    # Load employee data
    employee_data = pd.read_csv("Employee.csv",
                                 parse_dates=['HireDate'],
                                 date_format='%d.%m.%Y')

    # Load performance data
    performance_data = pd.read_csv("PerformanceRating.csv")

    return employee_data, performance_data

# Initial Data Exploration
def explore_data(data):
    print(data.head())
    print(data.info())
    print(data.describe())

def preprocess_data(employee_data, performance_data):
    # Merge performance data
    merged_data = pd.merge(employee_data, performance_data, on='EmployeeID', how='inner')

    # Define performance columns
    performance_columns = [
        'EnvironmentSatisfaction',
        'JobSatisfaction',
        'RelationshipSatisfaction',
        'WorkLifeBalance',
        'SelfRating',
        'ManagerRating'
    ]

    # Calculate comprehensive performance rating
    merged_data['PerformanceRating'] = merged_data[performance_columns].mean(axis=1)

    # Create binary performance classification
    merged_data['PerformanceClass'] = (
        merged_data['PerformanceRating'] > merged_data['PerformanceRating'].median()
    ).astype(int)

    # Numeric features
    numeric_features = [
        'Age',
        'DistanceFromHome (KM)',
        'Salary',
        'YearsAtCompany',
        'YearsInMostRecentRole',
        'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]

    # Categorical features
    categorical_features = [
        'Gender',
        'BusinessTravel',
        'Department',
        'State',
        'Ethnicity',
        'Education',
        'EducationField',
        'JobRole',
        'MaritalStatus',
        'OverTime'
    ]

    # Preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Separate features and targets for regression
    X_reg = merged_data[numeric_features + categorical_features]
    y_reg = merged_data['PerformanceRating']

    # Separate features and targets for classification
    X_class = merged_data[numeric_features + categorical_features]
    y_class = merged_data['PerformanceClass']

    # Split the data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

    return (
        merged_data,  # Return the merged DataFrame
        preprocessor,
        X_train_reg, X_test_reg, y_train_reg, y_test_reg,
        X_train_class, X_test_class, y_train_class, y_test_class
    )

def train_and_evaluate_regression_models(preprocessor, X_train, X_test, y_train, y_test):
    # Regression Pipelines
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store results
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        # Print results
        print(f"\n{name} Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared Score: {r2:.4f}")

    return results

def train_and_evaluate_classification_models(preprocessor, X_train, X_test, y_train, y_test):
    # Classification Pipelines
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        # Print results
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Confusion Matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    return results

def visualize_data(employee_data, performance_data, regression_results, classification_results):
    # Use a Matplotlib built-in style
    plt.style.use('default')

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Employee Performance Data Visualization', fontsize=16)

    # 1. Distribution of Performance Ratings
    performance_columns = [
        'EnvironmentSatisfaction',
        'JobSatisfaction',
        'RelationshipSatisfaction',
        'WorkLifeBalance',
        'SelfRating',
        'ManagerRating'
    ]

    performance_data[performance_columns].boxplot(ax=axs[0, 0])
    axs[0, 0].set_title('Distribution of Performance-Related Metrics')
    axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45, ha='right')

    # 2. Correlation Heatmap of Numeric Features
    numeric_features = [
        'Age',
        'DistanceFromHome (KM)',
        'Salary',
        'YearsAtCompany',
        'YearsInMostRecentRole',
        'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]

    # Correlation heatmap
    correlation_matrix = employee_data[numeric_features].corr()
    im = axs[0, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axs[0, 1].set_title('Correlation Heatmap of Numeric Features')
    axs[0, 1].set_xticks(np.arange(len(numeric_features)))
    axs[0, 1].set_yticks(np.arange(len(numeric_features)))
    axs[0, 1].set_xticklabels(numeric_features, rotation=45, ha='right')
    axs[0, 1].set_yticklabels(numeric_features)

    # Add correlation values to heatmap
    for i in range(len(numeric_features)):
        for j in range(len(numeric_features)):
            axs[0, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                            ha='center', va='center',
                            color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')

    # 3. Regression Model Performance Comparison
    model_names = list(regression_results.keys())
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']

    regression_data = []
    for model in model_names:
        regression_data.append([
            regression_results[model][metric] for metric in metrics
        ])

    x = np.arange(len(metrics))
    width = 0.35

    axs[1, 0].bar(x - width/2, regression_data[0], width, label=model_names[0])
    axs[1, 0].bar(x + width/2, regression_data[1], width, label=model_names[1])
    axs[1, 0].set_title('Regression Model Performance Comparison')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(metrics)
    axs[1, 0].legend()

    # 4. Classification Model Performance Comparison
    model_names = list(classification_results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    classification_data = []
    for model in model_names:
        classification_data.append([
            classification_results[model][metric] for metric in metrics
        ])

    x = np.arange(len(metrics))
    width = 0.35

    axs[1, 1].bar(x - width/2, classification_data[0], width, label=model_names[0])
    axs[1, 1].bar(x + width/2, classification_data[1], width, label=model_names[1])
    axs[1, 1].set_title('Classification Model Performance Comparison')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(metrics)
    axs[1, 1].legend()

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('performance_visualization.png')
    plt.close()

def main():
    # Load data
    employee_data, performance_data = load_data()

    # Explore Data
    print("Exploring Employee Data")
    explore_data(employee_data)

    print("Exploring Performance Data")
    explore_data(performance_data)

    # Preprocess data
    merged_data, preprocessor, X_train_reg, X_test_reg, y_train_reg, y_test_reg, \
    X_train_class, X_test_class, y_train_class, y_test_class = preprocess_data(
        employee_data, performance_data
    )

    # Train and evaluate regression models
    print("\nRegression Model Evaluation:")
    regression_results = train_and_evaluate_regression_models(
        preprocessor,
        X_train_reg, X_test_reg, y_train_reg, y_test_reg
    )

    # Train and evaluate classification models
    print("\nClassification Model Evaluation:")
    classification_results = train_and_evaluate_classification_models(
        preprocessor,
        X_train_class, X_test_class, y_train_class, y_test_class
    )

    # Generate visualizations
    visualize_data(employee_data, performance_data, regression_results, classification_results)

if __name__ == "__main__":
    main()