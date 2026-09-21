import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

'''
Returns selected features including positive, negative influenced features, 
where magnitude of selected features show the strong/weak association with target
'''

def lasso_CV(df_X, y, verbose=0):
    # ---- INPUTS ----
    # df_X: your feature matrix (DataFrame)
    # y: your continuous target variable (array)

    # Standardize features
    X_scaled = StandardScaler().fit_transform(df_X)

    # LASSO with 5-fold cross-validation
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)

    # Get coefficients
    lasso_coef = pd.Series(lasso.coef_, index=df_X.columns)

    # Filter non-zero features
    selected_features = lasso_coef[lasso_coef != 0].sort_values(ascending=False)

    if verbose:
        # Print top selected features (including positive, negative influenced features, 
        # where magnitude show the strong/weak association with target)
        print("Top selected features:")
        print(selected_features)

        if len(selected_features)!=0:
            plt.figure(figsize=(10, 6))
            selected_features.plot(kind='barh')
            plt.title("LASSO-Selected Features for Prediction")
            plt.xlabel("Coefficient")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
    top_feature_names = list(selected_features.index)
    top_features_wInfluence = selected_features
    return top_feature_names, top_features_wInfluence

## usage:
# top_feature_names, top_features_wInfluence = lasso_CV(df_X, y, verbose=0)