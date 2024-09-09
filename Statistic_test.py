
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# File path
file_path = r"D:\desktop_backup\Hot swappable UAV & Station\Pose_Estimation_Experiment_Results.xlsx"

# Read Excel file
df = pd.read_excel(file_path)

# Rename columns for easier use
df = df.rename(columns={
    'Camera Angle (degrees)': 'Camera_Angle',
    'Battery Position': 'Battery_Position',
    'Battery Orientation': 'Battery_Orientation',
    'Algorithm Model': 'Algorithm_Model',
    'Position MAE Overall': 'Position_MAE_Overall',
    'Position RMSE Overall': 'Position_RMSE_Overall',
    'Position SD Overall': 'Position_SD_Overall',
    'Attitude MAE Overall': 'Attitude_MAE_Overall',
    'Attitude RMSE Overall': 'Attitude_RMSE_Overall',
    'Attitude SD Overall': 'Attitude_SD_Overall'
})

# Data preprocessing
categorical_columns = ['Camera_Angle', 'Battery_Position', 'Battery_Orientation', 'Algorithm_Model']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Check and handle inf and NaN values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# 1. Block analysis
def block_analysis(data, angle):
    print(f"\nAnalysis for Camera Angle {angle}:")
    block_data = data[data['Camera_Angle'] == angle]
    metrics = ['Position_MAE_Overall', 'Position_RMSE_Overall', 'Position_SD_Overall',
               'Attitude_MAE_Overall', 'Attitude_RMSE_Overall', 'Attitude_SD_Overall']
    for metric in metrics:
        print(f"\n{metric}:")
        result = block_data.groupby(['Battery_Orientation', 'Algorithm_Model'])[metric].mean()
        print(result)
    return block_data

block_90 = block_analysis(df, 90)
block_45 = block_analysis(df, 45)

# 2. Kruskal-Wallis test
def perform_kruskal(data, dependent_var, factor):
    groups = [group for _, group in data.groupby(factor)[dependent_var]]
    h_statistic, p_value = stats.kruskal(*groups)
    print(f"Kruskal-Wallis test for {dependent_var} by {factor}:")
    print(f"H-statistic: {h_statistic:.4f}, p-value: {p_value:.4f}")

# Perform Kruskal-Wallis test for each factor
factors = ['Camera_Angle', 'Battery_Orientation', 'Algorithm_Model']
metrics = ['Position_MAE_Overall', 'Position_RMSE_Overall', 'Position_SD_Overall',
           'Attitude_MAE_Overall', 'Attitude_RMSE_Overall', 'Attitude_SD_Overall']
for metric in metrics:
    print(f"\nAnalysis for {metric}:")
    for factor in factors:
        perform_kruskal(df, metric, factor)

# 3. Visualization
output_dir = r"D:\desktop_backup\Hot swappable UAV & Station\analysis_output"
os.makedirs(output_dir, exist_ok=True)

def save_plot(fig, filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Figure saved to {filepath}")

# Box plots
for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Algorithm_Model', y=metric, hue='Camera_Angle', data=df, ax=ax)
    ax.set_title(f'{metric} by Algorithm Model and Camera Angle')
    save_plot(fig, f'{metric.lower()}_boxplot.png')

# 4. Model performance comparison
def model_performance_comparison(data, metric):
    models = data['Algorithm_Model'].unique()
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            stat, p = stats.mannwhitneyu(data[data['Algorithm_Model'] == model1][metric],
                                         data[data['Algorithm_Model'] == model2][metric])
            print(f"{metric} - {model1} vs {model2}: U-statistic = {stat:.4f}, p-value = {p:.4f}")

print("\nModel Performance Comparison:")
for metric in metrics:
    print(f"\n{metric}:")
    model_performance_comparison(df, metric)

# 5. Stability analysis
def stability_analysis(data):
    print("\nStability Analysis:")
    for model in data['Algorithm_Model'].unique():
        model_data = data[data['Algorithm_Model'] == model]
        for metric in metrics:
            metric_range = model_data[metric].max() - model_data[metric].min()
            print(f"{model} - {metric} Range: {metric_range:.6f}")

stability_analysis(df)

# 6. Best combination recommendation
def best_combination(data):
    print("\nBest Combinations:")
    for metric in metrics:
        best = data.loc[data[metric].idxmin()]
        print(f"Best for {metric}:")
        print(best[['Camera_Angle', 'Battery_Orientation', 'Algorithm_Model', metric]])
        print()

best_combination(df)

# 7. Heatmap visualization
for metric in ['Position_MAE_Overall', 'Position_RMSE_Overall', 'Attitude_MAE_Overall', 'Attitude_RMSE_Overall']:
    fig, ax = plt.subplots(figsize=(12, 10))
    pivot_table = df.pivot_table(values=metric, 
                                 index=['Camera_Angle'], 
                                 columns=['Battery_Orientation', 'Algorithm_Model'])
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.4f', ax=ax)
    ax.set_title(f'{metric} Heatmap')
    plt.tight_layout()
    save_plot(fig, f'{metric.lower()}_heatmap.png')

print("\nAnalysis complete. Check the generated plots in the output directory and console output for results.")
