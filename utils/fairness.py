import numpy as np
import pandas as pd

def demographic_parity_difference(y_true, y_pred, sensitive_attribute):
    # calculate positive prediction rate per group
    df = pd.DataFrame({
        'y_pred': y_pred,
        'sensitive': sensitive_attribute
    })
    
    rates = {}
    groups = df['sensitive'].unique()
    
    for group in groups:
        group_mask = df['sensitive'] == group
        if group_mask.sum() > 0:
            rates[group] = df.loc[group_mask, 'y_pred'].mean()
        else:
            rates[group] = 0.0
    
    # max difference between groups
    if len(rates) > 1:
        rate_values = list(rates.values())
        difference = max(rate_values) - min(rate_values)
    else:
        difference = 0.0
    
    return {
        'difference': difference,
        'rates': rates,
        'groups': groups.tolist()
    }


def disparate_impact_ratio(y_true, y_pred, sensitive_attribute, unprivileged_value, privileged_value):
    # ratio of positive prediction rates: unprivileged / privileged
    df = pd.DataFrame({
        'y_pred': y_pred,
        'sensitive': sensitive_attribute
    })
    
    unprivileged_mask = df['sensitive'] == unprivileged_value
    privileged_mask = df['sensitive'] == privileged_value
    
    unprivileged_rate = df.loc[unprivileged_mask, 'y_pred'].mean() if unprivileged_mask.sum() > 0 else 0.0
    privileged_rate = df.loc[privileged_mask, 'y_pred'].mean() if privileged_mask.sum() > 0 else 0.0
    
    if privileged_rate > 0:
        ratio = unprivileged_rate / privileged_rate
    else:
        ratio = 0.0 if unprivileged_rate == 0 else np.inf
    
    return {
        'ratio': ratio,
        'unprivileged_rate': unprivileged_rate,
        'privileged_rate': privileged_rate
    }


def equalized_odds_difference(y_true, y_pred, sensitive_attribute):
    # calculate tpr and fpr differences between groups
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive': sensitive_attribute
    })
    
    groups = df['sensitive'].unique()
    tpr_by_group = {}
    fpr_by_group = {}
    
    for group in groups:
        group_mask = df['sensitive'] == group
        group_df = df.loc[group_mask]
        
        if len(group_df) > 0:
            # true positive rate
            tp = ((group_df['y_true'] == 1) & (group_df['y_pred'] == 1)).sum()
            fn = ((group_df['y_true'] == 1) & (group_df['y_pred'] == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tpr_by_group[group] = tpr
            
            # false positive rate
            fp = ((group_df['y_true'] == 0) & (group_df['y_pred'] == 1)).sum()
            tn = ((group_df['y_true'] == 0) & (group_df['y_pred'] == 0)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fpr_by_group[group] = fpr
    
    tpr_values = list(tpr_by_group.values())
    fpr_values = list(fpr_by_group.values())
    
    tpr_difference = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0.0
    fpr_difference = max(fpr_values) - min(fpr_values) if len(fpr_values) > 1 else 0.0
    
    return {
        'tpr_difference': tpr_difference,
        'fpr_difference': fpr_difference,
        'tpr_by_group': tpr_by_group,
        'fpr_by_group': fpr_by_group
    }
