import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE


def evaluate_xgboost_model(X, y, features_list=None, n_splits=5, num_boost_round=100, random_state=42, smote_flag=True):
    # Auto-compute scale_pos_weight
    num_pos = np.sum(y == 1)
    num_neg = np.sum(y == 0)
    scale_pos_weight = num_neg / num_pos
    # print(f"[INFO] scale_pos_weight: {scale_pos_weight:.2f}")

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': num_boost_round,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'seed': random_state
    }

    # CV setup
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Tracking
    metrics = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']}
    roc_curve_data, pr_curve_data = [], []
    y_true_all, y_pred_prob_all = [], []

    best_model = None
    best_auc = 0

    fold = 0
    for train_idx, test_idx in kf.split(X, y):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if smote_flag:
            # Apply SMOTE to training data only
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

        y_pred_prob = model.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Store for global metrics
        y_true_all.extend(y_test)
        y_pred_prob_all.extend(y_pred_prob)

        # ROC and PR
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

        roc_curve_data.append((fpr, tpr))
        pr_curve_data.append((precision, recall))

        # Performance metrics
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        fold_roc_auc = auc(fpr, tpr)
        fold_pr_auc = average_precision_score(y_test, y_pred_prob)
        metrics['roc_auc'].append(fold_roc_auc)
        metrics['pr_auc'].append(fold_pr_auc)

        # Track best model
        if fold_roc_auc > best_auc:
            best_auc = fold_roc_auc
            best_model = model
            best_fold_data = (X_test, y_test)  # for SHAP later

        # print(f"[Fold {fold}] ROC AUC: {fold_roc_auc:.3f} | PR AUC: {fold_pr_auc:.3f}")

    # ---- Print Metric Summary ----
    print("\n[Summary] Cross-Validated Performance Metrics (mean ± std):")
    for key in metrics:
        print(f"{key:10s}: {np.mean(metrics[key]):.3f} ± {np.std(metrics[key]):.3f}")

    # ---- Plot ROC and PR Curves with CI ----
    mean_fpr = np.linspace(0, 1, 100)
    tpr_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curve_data]
    pr_interp = [np.interp(mean_fpr, recall[::-1], precision[::-1]) for precision, recall in pr_curve_data]
    mean_tpr = np.mean(tpr_interp, axis=0)
    mean_precision = np.mean(pr_interp, axis=0)

    tpr_lower = np.percentile(tpr_interp, 2.5, axis=0)
    tpr_upper = np.percentile(tpr_interp, 97.5, axis=0)
    pr_lower = np.percentile(pr_interp, 2.5, axis=0)
    pr_upper = np.percentile(pr_interp, 97.5, axis=0)

    roc_auc = auc(mean_fpr, mean_tpr)
    pr_auc = average_precision_score(y_true_all, y_pred_prob_all)

    plt.figure(figsize=(10, 4))
    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {roc_auc:.2f})')
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2, label='95% CI')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # PR
    plt.subplot(1, 2, 2)
    plt.plot(mean_fpr, mean_precision, color='green', label=f'Mean PR (AUC = {pr_auc:.2f})')
    plt.fill_between(mean_fpr, pr_lower, pr_upper, color='green', alpha=0.2, label='95% CI')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

    
    # ---- SHAP Beeswarm Plot for Best Model ----
    print("\n[SHAP] Generating SHAP beeswarm plot for best fold...")
    X_best = best_fold_data[0]
    if features_list is None:
        X_best_df = X_best
    else:
        X_best_df = pd.DataFrame(X_best, columns = features_list)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_best_df)
    shap.summary_plot(shap_values, features=X_best_df)
    # shap.plots.beeswarm(shap_values)

    return best_model
