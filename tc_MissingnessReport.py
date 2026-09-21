import pandas as pd

def missingness_report(df, thresholds=[0.3, 0.6, 1.0]):
    """
    Print a report of missing value percentages and risk flags per column.
    
    Parameters:
    df : pd.DataFrame
        The input DataFrame with potential missing values.
    thresholds : list of float
        Thresholds to categorize risk levels (default: [30%, 60%, 100%]).
    """
    missing_percent = df.isnull().mean()
    report = pd.DataFrame({
        'Missing %': (missing_percent * 100).round(2),
        'Risk Level': pd.cut(missing_percent,
                             bins=[-0.01, thresholds[0], thresholds[1], thresholds[2]],
                             labels=['Low (✔️)', 'Moderate (⚠️)', 'High (🚫)'])
    }).sort_values('Missing %', ascending=False)

    print("🧪 Missingness Report by Column:\n")
    print(report)
    return report




# ### USE-CASE
# # Example DataFrame with artificial missingness
# df = pd.DataFrame({
#     'heart_rate': [75, 80, np.nan, 70, 65, np.nan, 90],
#     'resp_rate': [np.nan]*7,
#     'spo2': [98, 97, 99, np.nan, 96, 95, 94],
#     'age': [60, 55, 70, 65, np.nan, 75, 68],
#     'lactate': [np.nan, 1.2, 1.1, np.nan, np.nan, 2.0, 1.4]
# })

# report = missingness_report(df)
# df