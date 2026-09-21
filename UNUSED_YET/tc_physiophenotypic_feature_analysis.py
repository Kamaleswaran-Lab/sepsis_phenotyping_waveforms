'''
# Phenotypic Feature Extraction

- Frequency of phenotype-memberships, 
- their cross transition probabilities, and 
- their occurrence intervals (Self transition probabilities)
- Dwell times
- Entropy
- Lag of first appearance
'''

#General + Feature extraction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import entropy

#Classification
import UNUSED_YET.tc_xgboostCV_imbalancedBinaryClass_SHAP as tc_xgb_BinClass
import scikitplot as skplt
#from sklearn.metrics import (classification_report, accuracy_score, roc_curve, auc, precision_recall_curve)


#Regression
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import re
import smogn

#Feature selection
import seaborn as sns
from UNUSED_YET.tc_LASSO_feature_importance import lasso_CV


def tc_phenotypic_feature_extraction(df, block_duration):
    '''
    INPUTS:
    df: patient encounter table having phenotype memberships for each 5min segment in columns
    block_duration: Duration of block for feature extraction ('15min', '30min', '1hr', '3hr', '6hr')
    
    OUTPUTS:
    df_combined with following columns wrt phenotypes---
    FreqN: normalized frequedncy/repeatation of phenotypes for the block analyzed 
    CTP: cross transition probabilities
    STP: self occurrence probabilities
    Dwell time: max consecutive runs
    Entropy: disorder of phenotype sequence
    Lag: index of first appearance (normalized)
    '''
    total_segments = {'15min':3, '30min':6, '1hr':12, '3hr': 36, '6hr': 72} #each segment size is 5min
    num = total_segments[block_duration]
    col_id0 = list(df.columns).index('label_seg0') #column number of 1st presepsis segment  
    df_small = df.iloc[:,col_id0:col_id0+num].copy()
    n_phenotypes = df.iloc[:,col_id0].nunique()
    
    # Example row: [3,1,1,3,NaN,NaN] has normalized frequencies- 0:0, 1:2/4, 2:0, 3:2/4
    def tc_RowPhenotypeCount(row, n):
        count = np.zeros(n)
        row = row.dropna()
        # Only keep valid integers within range
        row = row[row.apply(lambda x: isinstance(x, (int, float)) and 0 <= x < n)]
        # Count normalized frequencies
        value_counts = row.value_counts(normalize=True)
        for val, freq in value_counts.items():
            count[int(val)] = freq
        return count
    # Output column names
    freq_cols = [f'{block_duration}_FreqN_{i}' for i in range(n_phenotypes)]
    # Apply row-wise frequency count function
    df[freq_cols] = df_small.apply(lambda row: pd.Series(tc_RowPhenotypeCount(row, n_phenotypes)), axis=1)
    # print(df[freq_cols])
    
    
    # Define all transitions
    all_transitions = [f'{block_duration}_CTP_{i}-{j}' for i, j in product(range(n_phenotypes), repeat=2)]
    self_transitions_all = [f'{block_duration}_CTP_{i}-{i}' for i in range(n_phenotypes)]
    cross_transitions = sorted(set(all_transitions) - set(self_transitions_all))

    # Function to compute cross transition probabilities only
    # Example row: [3,1,1,3,NaN,NaN] has total cross transitions of 2 (not 3). Probs= [1-3]:1/2, [3-1]-1/2
    def tc_cross_transition_probs(row):
        cross_trans_counts = dict.fromkeys(cross_transitions, 0.0)
        row = row.dropna().astype(int).values
        valid_transitions = 0
        for i in range(len(row) - 1):
            from_state, to_state = row[i], row[i + 1]
            if from_state != to_state:  # Cross transition only
                key = f"{block_duration}_CTP_{from_state}-{to_state}"
                if key in cross_trans_counts:
                    cross_trans_counts[key] += 1
                    valid_transitions += 1
        if valid_transitions > 0:
            for key in cross_trans_counts:
                cross_trans_counts[key] /= valid_transitions
        return pd.Series(cross_trans_counts)
    df_cross_probs = df_small.apply(tc_cross_transition_probs, axis=1)
    
    
    def tc_self_transition_probs(row):
        self_trans_counts = {f"{block_duration}_STP_{i}-{i}": 0.0 for i in range(n_phenotypes)}
        row = row.dropna().astype(int).values
        if len(row) < 2:
            return pd.Series(self_trans_counts)

        total_self_transitions = 0
        for i in range(len(row) - 1):
            from_state, to_state = row[i], row[i + 1]
            if from_state == to_state:
                key = f"{block_duration}_STP_{from_state}-{to_state}"
                if key in self_trans_counts:
                    self_trans_counts[key] += 1
                    total_self_transitions += 1

        if total_self_transitions > 0:
            for key in self_trans_counts:
                self_trans_counts[key] /= total_self_transitions
        return pd.Series(self_trans_counts)
    df_self_probs = df_small.apply(tc_self_transition_probs, axis=1)
    
    # New engineered features: dwell time, entropy, lag
    def phenotype_sequence_features(row):
        row = row.dropna().astype(int).tolist()
        feats = {}
        for i in range(n_phenotypes):
            label = f'{block_duration}_Dwell_{i}'
            max_run = curr = 0
            for x in row:
                if x == i:
                    curr += 1
                    max_run = max(max_run, curr)
                else:
                    curr = 0
            feats[label] = max_run / len(row) if len(row) else 0.0
        
        counts = np.bincount(row, minlength=n_phenotypes)
        feats[f'{block_duration}_Ent'] = entropy(counts) if len(row) else 0.0

        for i in range(n_phenotypes):
            label = f'{block_duration}_LFA_{i}'
            try:
                feats[label] = row.index(i) / len(row)
            except ValueError:
                feats[label] = 1.0
        return pd.Series(feats)

    df_seq_feats = df_small.apply(phenotype_sequence_features, axis=1)
    
    
    # Combine with original dataframe
    df_combined = pd.concat([df, df_cross_probs, df_self_probs, df_seq_feats], axis=1)
    df_combined.reset_index(drop=True, inplace=True)
    return df_combined

# Example row: [3,1,1,3,NaN,NaN] has normalized frequencies- 0:0, 1:2/4, 2:0, 3:2/4
# Example row: [3,1,1,3,NaN,NaN] has total cross transitions of 2 (not 3). Probs= [1-3]:1/2, [3-1]-1/2
# Example row1: [3,1,1,3,NaN,NaN] has occurrence probabilities: 0-0:0, 1-1:1/1, 2-2:0, 3-3:0
# Example row2: [3,1,1,1,3,3] has occurrence probabilities: 0-0:0, 1-1:2/5, 2-2:0, 3-3:1/5

'''
# Just for testing purpose
df_test = pd.DataFrame({
    'label_seg0': [3,0, 1, 2, 1, 3],
    'label_seg1': [1,0, 1, 2, 2, 3],
    'label_seg2': [1,1, 1, 2, 1, 0],
    'label_seg3': [3,1, 2, 3, 3, 0],
    'label_seg4': [np.nan,2, 2, 3, 3, 1],
    'label_seg5': [np.nan,2, np.nan, 3, np.nan, 1]})
df_out = tc_phenotypic_feature_extraction(df_test.copy(), block_duration='30min')
df_out

'''


### Short-term outcome classification from physiophenotypic features
def XGBmodel_phenotype_features_STO_classification(df, eventXdays_postS3onset, feature_win = ('label_seg71','Death28')): 
    # ==== for short-term mortality (0/1) and septic shock (0/1): binary class classification === #
    ### eventXdays_postS3onset: target column name ('Death28','Death7','Death2','Death1','ss28','ss7','ss2','ss1')
    # --- Feature Extraction
    col_id0 = list(df.columns).index(feature_win[0]) #column number of last presepsis segment 
    col_id1 = list(df.columns).index(feature_win[1]) #column number of 1st target variable
    df_feature = df.iloc[:,col_id0+1:col_id1].copy()
    y = df[eventXdays_postS3onset].values
    # --- Feature Selection
    df_feat_extend = pd.concat([df_feature, df[eventXdays_postS3onset]], axis=1)
    corrx = df_feat_extend.corr()
    f, ax = plt.subplots(figsize=(10,9))
    cmap0 = sns.diverging_palette(220, 10, as_cmap=True)
    heatmap0 = sns.heatmap(corrx, cmap=cmap0, center=0.0, vmax=1, linewidth=1, ax=ax)
    top_feature_names, top_features_wInfluence = lasso_CV(df_feature, y, verbose=1) #LASSO Feature Selection
    if len(top_feature_names)>1:
        print('================== FS: With LASSO SELECTED FEATURES ==================')
        X = df_feature[top_feature_names].values
        best_model = tc_xgb_BinClass.evaluate_xgboost_model(X, y, features_list=top_feature_names)
    else:
        print('================== FS: SINCE NO INFLUENCING FEATURES, SO CONSIDERING ALL FEATURES ==================')
        X = df_feature.values
        best_model = tc_xgb_BinClass.evaluate_xgboost_model(X, y, features_list=list(df_feature.columns))
    return best_model

def Block_STO_prediction_phenotypicFeature(df, block_duration, eventXdays_postS3onset, df_origin, feature_win = ('label_seg71','Death28')): 
    #block_duration: '15min', '1hr'...
    ### eventXdays_postS3onset: target column name ('Death28','Death7','Death2','Death1','ss28','ss7','ss2','ss1')
    print(f'================== PREDICTION OF {eventXdays_postS3onset} WITH {block_duration} DATA ==================')
    df_allFIN_feature = tc_phenotypic_feature_extraction(df, block_duration)
    #df_allFIN_feature['studyID'] = df_allFIN_feature['studyID'].astype('str')
    #df_origin['studyID'] = df_origin['studyID'].astype('str')
    df_allFIN_feature['fin'] = df_allFIN_feature['fin'].astype('int')
    df_origin['fin'] = df_origin['fin'].astype('int')
    colz = ['fin','Death28','Death7','Death2','Death1','ss28','ss7','ss2','ss1'] #'studyID',
    df_feature_wTarget = df_allFIN_feature.merge(df_origin[colz].drop_duplicates(), how='left', on=['fin']).drop_duplicates()
    df_feature_wTarget.reset_index(inplace=True, drop=True) 
    _ = XGBmodel_phenotype_features_STO_classification(df_feature_wTarget, eventXdays_postS3onset, feature_win)
    return df_feature_wTarget

def Block_STO_prediction_anyFeature(df, eventXdays_postS3onset, feature_win = ('x','y')): 
    #block_duration: '15min', '1hr'...
    ### eventXdays_postS3onset: target column name ('Death28','Death7','Death2','Death1','ss28','ss7','ss2','ss1')
    print(f'================== PREDICTION OF {eventXdays_postS3onset} WITH FEATURE DATA ==================')
    
    df['fin'] = df['fin'].astype('int')
    colz = ['fin','Death28','Death7','Death2','Death1','ss28','ss7','ss2','ss1'] #'studyID',
    df_feature_wTarget = df.drop_duplicates()
    df_feature_wTarget.reset_index(inplace=True, drop=True) 
    _ = XGBmodel_phenotype_features_STO_classification(df_feature_wTarget, eventXdays_postS3onset, feature_win)
    return df_feature_wTarget
        




def XGBmodel_phenotype_features_regression(df, target, block_duration, feature_win = ('label_seg71','Death28'), smogn=False):
    #e.g. target = 'vpfd28'
    letters = re.findall(r'[a-zA-Z]+', target)
    numbers = re.findall(r'\d+', target)
    outcome_name = ''.join(letters).upper()
    Xdays_postS3onset = int(numbers[0]) if numbers else None
    print(f'============== PREDICTION OF {Xdays_postS3onset}-DAY {outcome_name} WITH {block_duration} DATA ==============')
    
    # --- Target ---
    try:
        y = df[target].values
    except:
        raise ValueError('Invalid target name! Use one of: "vpfdX", "vfdX", "bfdX", where X belongs to 28 / 8 / 2')

    # --- Feature Extraction ---
    col_id0 = list(df.columns).index(feature_win[0])
    col_id1 = list(df.columns).index(feature_win[1])
    df_feature = df.iloc[:, col_id0 + 1:col_id1].copy()
    
    # --- Feature Selection ---
    df_feat_extend = pd.concat([df_feature, df[[target]]], axis=1)
    corrx = df_feat_extend.corr()
    f, ax = plt.subplots(figsize=(10,9))
    cmap0 = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure()
    heatmap0 = sns.heatmap(corrx, cmap=cmap0, center=0.0, vmax=1, linewidth=1, ax=ax)
    
    top_feature_names, top_features_wInfluence = lasso_CV(df_feature, y, verbose=1) 
    if len(top_feature_names)>1:
        print('================== FS: With LASSO SELECTED FEATURES ==================')
        X0 = df_feature[top_feature_names]
    else:
        print('================== FS: SINCE NO INFLUENCING FEATURES, SO CONSIDERING ALL FEATURES ==================')
        X0 = df_feature
    
    df_selected = pd.concat([X0, df[[target]]], axis=1)
    
    # Resample using SMOGN (handles imbalanced regression targets)
    if smogn:
        df_resampled = smogn.smoter(
            data=df_selected,
            y=target,            # Your target column name
            samp_method='balance'  # 'extreme' or 'balance'
        )
        # Separate features and target
        X = df_resampled.drop(columns=[target]).values
    else:
        X = X0.values
    
    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # --- Convert to DMatrix ---
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # --- XGBoost Parameters ---
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'rmse',
        'seed': 42
    }

    # --- Train the Model ---
    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=10,
        evals_result=evals_result,
        verbose_eval=False
    )

    # --- Predictions ---
    y_pred = model.predict(dtest)

    # --- Evaluation Metrics ---
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    corr_coeff, _ = pearsonr(y_test, y_pred)
    print(f"MAE: {mae:.2f}, R²: {r2:.2f}, Pearson Corr: {corr_coeff:.2f}")

    # --- Bland-Altman Plot Data ---
    avg = (y_test + y_pred) / 2
    diff = y_test - y_pred
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # --- SHAP (TreeExplainer) ---
    print("[SHAP] Computing SHAP values...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    plt.figure()
    shap.summary_plot(shap_values, features=X_test, feature_names=df_feature.columns)

    # --- Plots ---
    plt.figure(figsize=(10, 4))

    # A. Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Correlation Plot ({block_duration})")
    plt.legend(loc='upper left', title=f"MAE = {mae:.2f}\nR² = {r2:.2f}\nPCorr = {corr_coeff:.2f}")

    # B. Bland-Altman
    plt.subplot(1, 2, 2)
    plt.scatter(avg, diff, alpha=0.5, color='darkorange')
    plt.axhline(mean_diff, color='black', linestyle='--', label=f"Mean Diff = {mean_diff:.2f}")
    plt.axhline(loa_upper, color='red', linestyle='--', label=f"Upper LoA = {loa_upper:.2f}")
    plt.axhline(loa_lower, color='red', linestyle='--', label=f"Lower LoA = {loa_lower:.2f}")
    plt.title(f"Bland-Altman Plot ({block_duration})")
    plt.xlabel("Average of Actual & Predicted")
    plt.ylabel("Difference (Actual - Predicted)")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return model


def Block_event_regression_phenotypicFeature(df, block_duration, event_name, df_origin): 
    # block_duration: '15min', '1hr'...
    # event_name: 'vpfdX', 'vfdX', 'bfdX'
    df_allFIN_feature = tc_phenotypic_feature_extraction(df, block_duration)
    #df_allFIN_feature['studyID'] = df_allFIN_feature['studyID'].astype('str')
    #df_origin['studyID'] = df_origin['studyID'].astype('str')
    df_allFIN_feature['fin'] = df_allFIN_feature['fin'].astype('int')
    df_origin['fin'] = df_origin['fin'].astype('int')
    colz = ['fin','Death28','vpfd28','vpfd7','vpfd2','vfd28','vfd7','vfd2','bfd28','bfd7','bfd2'] #'studyID',
    df_feature_wTarget = df_allFIN_feature.merge(df_origin[colz].drop_duplicates(), how='left', on=['fin']).drop_duplicates()
    df_feature_wTarget.reset_index(inplace=True, drop=True)
    _ = XGBmodel_phenotype_features_regression(df_feature_wTarget, target=event_name, block_duration=block_duration)
    return df_feature_wTarget

def Block_event_regression_anyFeature(df, event_name, feature_win = ('x','y')): 
    block_duration= 0
    # event_name: 'vpfdX', 'vfdX', 'bfdX'
    df['fin'] = df['fin'].astype('int')
    colz = ['fin','Death28','vpfd28','vpfd7','vpfd2','vfd28','vfd7','vfd2','bfd28','bfd7','bfd2'] #'studyID',
    df = df.drop_duplicates()
    df.reset_index(inplace=True, drop=True)
    _ = XGBmodel_phenotype_features_regression(df, target=event_name, block_duration=block_duration, feature_win=feature_win)
    return df












# import xgboost as xgb
# import shap
# def XGBmodel_phenotype_features_regression(df, target, block_duration): #For regression of VFD, VPFD, BFD
#     col_id0 = list(df.columns).index('label_seg71') #column number of last presepsis segment 
#     col_id1 = list(df.columns).index('Death28') #column number of 1st target variable
#     df_feature = df.iloc[:,col_id0+1:col_id1].copy()
#     X = df_feature.values
#     if target=='vpfd': y = df['Vasopressor-free days'].values
#     elif target=='vfd': y = df['Ventilator-free days'].values
#     elif target=='bfd': y = df['Bolus-free days'].values
#     else: print('Invalid target name!')
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
#     model = xgb.XGBRegressor(
#         objective='reg:squarederror',
#         n_estimators=100,
#         max_depth=4,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )
#     evals_result={}
#     model.fit(X_train, y_train, 
#               early_stopping_rounds=10, 
#               eval_set=[(X_train, y_train),(X_test, y_test)], 
#               evals_result=evals_result, verbose=True)

#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_val, preds)
#     rmse = mean_squared_error(y_val, preds, squared=False)
#     r2 = r2_score(y_test, y_pred)
#     corr_coeff, _ = pearsonr(y_test, y_pred)
#     print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, r2: {r2:.2f}, PCorr_coef: {corr_coeff:.2f}")

#     # Bland-Altman calculations
#     avg = (y_test + y_pred) / 2
#     diff = y_test - y_pred
#     mean_diff = np.mean(diff)
#     std_diff = np.std(diff)
#     loa_upper = mean_diff + 1.96 * std_diff
#     loa_lower = mean_diff - 1.96 * std_diff
    
#     # SHAP for explainability
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer(X_test)

#     # Summary plot (global importance)
#     shap.summary_plot(shap_values, X_test)

#     # Optional: Force plot for a single patient
#     shap.plots.force(shap_values[0])  # Uncomment to visualize individual prediction
    

#     # --- PLOTS ---
#     plt.figure(figsize=(10, 4))

#     # A. Actual vs Predicted
#     plt.subplot(1, 2, 1)
#     plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', label='Predictions')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
#     plt.xlabel("Actual Values")
#     plt.ylabel("Predicted Values")
#     plt.title(f"Correlation Plot ({block_duration})")

#     # Add legend with metrics
#     metrics_text = f"MSE = {mse:.2f}\nR² = {r2:.2f}\nPearson Corr = {corr_coeff:.2f}"
#     plt.legend(loc='upper left', title=metrics_text)

#     # B. Bland-Altman Plot
#     plt.subplot(1, 2, 2)
#     plt.scatter(avg, diff, alpha=0.5, color='darkorange')
#     plt.axhline(mean_diff, color='black', linestyle='--', label=f"Mean Diff = {mean_diff:.2f}")
#     plt.axhline(loa_upper, color='red', linestyle='--', label=f"Upper LoA = {loa_upper:.2f}")
#     plt.axhline(loa_lower, color='red', linestyle='--', label=f"Lower LoA = {loa_lower:.2f}")
#     plt.title("Bland-Altman Plot ({block_duration})")
#     plt.xlabel("Average of Actual & Predicted")
#     plt.ylabel("Difference (Actual - Predicted)")
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()
#     return y_pred



# import xgboost as xgb
# import shap
# def XGBmodel_phenotype_features_STM_classification(df, block_duration): #for short-term mortality binary class classification
#     col_id0 = list(df.columns).index('label_seg71') #column number of last presepsis segment 
#     col_id1 = list(df.columns).index('Death28') #column number of 1st target variable
#     df_feature = df.iloc[:,col_id0+1:col_id1].copy()
#     X = df_feature.values
#     y = df['Death28'].values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
#     # print('len(y_train), len(y_test)', len(y_train), len(y_test))
#     neg, pos = np.sum(y_train == 0), np.sum(y_train == 1) # Calculate the scale_pos_weight
#     class_weight = neg / pos
#     evals_result={}
    
#     model = xgb.XGBClassifier(
#         objective='binary:logistic',     # Binary classification task
#         random_state=42,
#         use_label_encoder=False,         # Disable legacy label encoder (avoid warnings)
#         scale_pos_weight=class_weight    # Handle class imbalance
#     )

#     model.fit(
#         X_train, y_train,
#         eval_metric=['auc', 'logloss'],                      # Track AUC and logloss
#         eval_set=[(X_train, y_train), (X_test, y_test)],     # Monitor both train and test
#         early_stopping_rounds=10,                            # Stop if test score doesn't improve
#         evals_result=evals_result,                           # Save training history
#         verbose=True                                         # Print training progress
#     )
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1] #probabilities for class 1
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     roc_auc = auc(fpr, tpr)
    
#     # Use TreeExplainer for XGBoost
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_train)
#     # SHAP Summary Plot
#     shap.summary_plot(shap_values, X_train)
    
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='blue')
#     plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(f"ROC Curve ({block_duration})")
#     plt.legend()
    
#     precision, recall, _ = precision_recall_curve(y_test, y_proba)
#     plt.subplot(1, 2, 2)
#     plt.plot(recall, precision, label="PR Curve", color='green')
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(f"Precision-Recall Curve ({block_duration})")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()
#     return y_pred