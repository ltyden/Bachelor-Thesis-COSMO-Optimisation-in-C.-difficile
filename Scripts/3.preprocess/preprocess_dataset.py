#!/usr/bin/env python3
"""
Dataset Preprocessing Script for Regression Analysis

This script performs comprehensive preprocessing on your COSMO optimization dataset
following the checklist: deduplication, missing value handling, variance checking,
outlier detection, multicollinearity analysis, train-test split, and feature scaling.

Output: cleaned_dataset.csv (preprocessed data for regression)
        train_set.csv, test_set.csv (80/20 split if save_splits=True)
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANALYSIS_DIR = PROJECT_ROOT / "parameter_optimisation" / "analysis"
CLEANED_DIR = PROJECT_ROOT / "parameter_optimisation" / "cleaned_datasets"


def load_dataset(input_file):
    print(f"\n{'='*70}")
    print(f"Loading dataset from: {input_file}")
    print(f"{'='*70}")

    if not Path(input_file).exists():
        print(f"ERROR: File {input_file} not found!")
        sys.exit(1)

    df = pd.read_csv(input_file, sep=';')
    print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    return df


def step1_remove_duplicates(df, subset_cols):
    print(f"\n{'='*70}")
    print("STEP 1: REMOVING DUPLICATE ROWS")
    print(f"{'='*70}")

    rows_before = len(df)
    df = df.drop_duplicates(subset=subset_cols)
    rows_after = len(df)

    duplicates_removed = rows_before - rows_after
    print(f"Rows before deduplication: {rows_before}")
    print(f"Rows after deduplication:  {rows_after}")
    print(f"Duplicates removed:        {duplicates_removed}")

    if duplicates_removed > 0:
        print(f"⚠ WARNING: Found {duplicates_removed} duplicate parameter combinations")
    else:
        print(f"✓ No duplicates found")

    return df.reset_index(drop=True)


def step2_missing_values(df):
    print(f"\n{'='*70}")
    print("STEP 2: CHECKING FOR MISSING VALUES")
    print(f"{'='*70}")

    missing_count = df.isnull().sum()
    print("Missing values per column:")
    print(missing_count)

    total_missing = missing_count.sum()
    if total_missing > 0:
        print(f"\n⚠ WARNING: Found {total_missing} missing values")
        missing_rows = df.isnull().any(axis=1).sum()
        print(f"Rows with at least one missing value: {missing_rows}")
        print(f"Dropping rows with missing values...")

        rows_before = len(df)
        df = df.dropna()
        print(f"Rows removed: {rows_before - len(df)}")
    else:
        print(f"✓ No missing values found")

    return df.reset_index(drop=True)


def step3_zero_variance(df, param_cols):
    print(f"\n{'='*70}")
    print("STEP 3: CHECKING FOR ZERO/NEAR-ZERO VARIANCE IN PARAMETERS")
    print(f"{'='*70}")

    print("\nParameter statistics:")
    print(df[param_cols].describe())

    print("\nVariance per parameter:")
    for col in param_cols:
        variance = df[col].var()
        std_dev = df[col].std()
        unique_count = df[col].nunique()
        print(f"  {col}: variance={variance:.6f}, std_dev={std_dev:.6f}, unique_values={unique_count}")
        if variance < 1e-6:
            print(f"  ⚠ WARNING: {col} has near-zero variance — consider dropping it")

    print(f"✓ Variance check complete")


def step4_outcome_distribution(df, outcome_col, save_plots=True):
    print(f"\n{'='*70}")
    print(f"STEP 4: CHECKING OUTCOME VARIABLE ({outcome_col}) DISTRIBUTION")
    print(f"{'='*70}")

    print(f"\n{outcome_col} Score Statistics:")
    print(df[outcome_col].describe())

    zeros = (df[outcome_col] == 0).sum()
    print(f"\nRuns with {outcome_col} = 0: {zeros} ({100*zeros/len(df):.1f}%)")

    if save_plots:
        plt.figure(figsize=(10, 6))
        plt.hist(df[outcome_col], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel(f'{outcome_col} Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {outcome_col} Scores Across Parameter Combinations')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ANALYSIS_DIR / f'{outcome_col.lower()}_distribution.png'
        plt.savefig(out_path, dpi=300)
        print(f"\n✓ Distribution plot saved to: {out_path}")
        plt.close()


def step5_handle_outliers(df, outcome_col):
    print(f"\n{'='*70}")
    print("STEP 5: IDENTIFYING OUTLIERS IN OUTCOME VARIABLE")
    print(f"{'='*70}")

    Q1 = df[outcome_col].quantile(0.25)
    Q3 = df[outcome_col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Q1 (25th percentile): {Q1:.4f}")
    print(f"Q3 (75th percentile): {Q3:.4f}")
    print(f"IQR: {IQR:.4f}")
    print(f"Outlier bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")

    outlier_mask = (df[outcome_col] < lower_bound) | (df[outcome_col] > upper_bound)
    print(f"\nOutliers detected (IQR method): {outlier_mask.sum()} ({100*outlier_mask.mean():.1f}%)")

    df = df.copy()
    df['is_outlier'] = outlier_mask.astype(int)
    df[f'{outcome_col}_is_zero'] = (df[outcome_col] == 0).astype(int)

    print(f"✓ Created 'is_outlier' column to flag outliers (0=normal, 1=outlier)")
    print(f"✓ Created '{outcome_col}_is_zero' column to flag zero runs (0=normal, 1=zero)")

    return df


def step6_multicollinearity(df, param_cols, save_plots=True):
    print(f"\n{'='*70}")
    print("STEP 6: CHECKING FOR MULTICOLLINEARITY")
    print(f"{'='*70}")

    corr_matrix = df[param_cols].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(4))

    if save_plots:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ANALYSIS_DIR / 'correlation_matrix.png'
        plt.savefig(out_path, dpi=300)
        print(f"\n✓ Correlation plot saved to: {out_path}")
        plt.close()

    print("\nVariance Inflation Factor (VIF) Scores:")
    X = df[param_cols].values
    vif_data = pd.DataFrame({
        'Parameter': param_cols,
        'VIF': [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    })
    print(vif_data.to_string(index=False))

    max_vif = vif_data['VIF'].max()
    if max_vif > 10:
        print(f"\n⚠ WARNING: Maximum VIF = {max_vif:.2f} > 10 (SEVERE multicollinearity)")
        print("  Consider: Ridge Regression, remove highly correlated features, or PCA")
    elif max_vif > 5:
        print(f"\n⚠ WARNING: Maximum VIF = {max_vif:.2f} > 5 (Moderate multicollinearity)")
        print("  Consider: Ridge Regression")
    else:
        print(f"\n✓ VIF scores all < 5 (low multicollinearity)")


def step7_train_test_split(df, param_cols, outcome_col, test_size=0.2, random_state=42):
    print(f"\n{'='*70}")
    print("STEP 7: TRAIN-TEST SPLIT")
    print(f"{'='*70}")

    X = df[param_cols]
    y = df[outcome_col]

    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Test size ratio: {test_size*100:.0f}%")
    print(f"Training set: {len(X_train)} rows ({100*len(X_train)/len(df):.0f}%)")
    print(f"Test set:     {len(X_test)} rows ({100*len(X_test)/len(df):.0f}%)")

    df_train = df.loc[X_train.index].reset_index(drop=True)
    df_test = df.loc[X_test.index].reset_index(drop=True)

    return df_train, df_test


def step8_feature_scaling(df_train, df_test, param_cols, scaling_method='minmax'):
    """Fit scaler on training set only, then apply to both sets to avoid data leakage."""
    print(f"\n{'='*70}")
    print("STEP 8: FEATURE SCALING")
    print(f"{'='*70}")

    print(f"Scaling method: {scaling_method.upper()}")

    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
        print("Scaling to [0, 1] range (Min-Max normalization)")
    else:
        scaler = StandardScaler()
        print("Scaling to mean=0, std=1 (Standard scaling)")

    scaler.fit(df_train[param_cols])

    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[param_cols] = scaler.transform(df_train[param_cols])
    df_test_scaled[param_cols] = scaler.transform(df_test[param_cols])

    scaler_path = CLEANED_DIR / f'scaler_{scaling_method}.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"\n✓ Scaler fitted on training set only (no leakage)")
    print(f"✓ Scaler saved to: {scaler_path}")

    print("\nScaled training parameter statistics:")
    print(df_train_scaled[param_cols].describe())

    return df_train_scaled, df_test_scaled, scaler


def step9_coverage_check(df, param_cols):
    print(f"\n{'='*70}")
    print("STEP 9: PARAMETER COVERAGE CHECK (unscaled)")
    print(f"{'='*70}")

    print("\nParameter ranges across full dataset:")
    for col in param_cols:
        print(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")

    print("\n✓ Parameter space coverage verified")


def main():
    valid_outcomes = ['TP%', 'TP', 'FN', 'FP', 'Precision', 'Recall', 'F1']

    parser = argparse.ArgumentParser(
        description='Preprocess regression dataset for COSMO optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Valid outcome variables: {", ".join(valid_outcomes)}

Examples:
  python preprocess_dataset.py raw_dataset_regression.csv F1 --scaling minmax
  python preprocess_dataset.py raw_dataset_regression.csv Precision --scaling standard
  python preprocess_dataset.py raw_dataset_regression.csv Recall -s minmax
        '''
    )

    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument(
        'outcome_variable',
        choices=valid_outcomes,
        help=f'Dependent variable (Y-variable) to predict'
    )
    parser.add_argument(
        '--scaling', '-s',
        choices=['minmax', 'standard'],
        default='minmax',
        help='Feature scaling method (default: minmax)'
    )
    parser.add_argument(
        '--output', '-o',
        default='cleaned_dataset.csv',
        help='Output filename (saved under parameter_optimisation/cleaned_datasets/)'
    )
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--no-splits', action='store_true', help='Skip train/test split files')

    args = parser.parse_args()

    param_cols = ['CDS_min', 'IGR_min', 'FD_CDS-CDS_min', 'FD_IGR-CDS_min']
    input_file = args.input_file
    output_file = CLEANED_DIR / Path(args.output).name
    outcome_col = args.outcome_variable
    scaling_method = args.scaling
    save_plots = not args.no_plots
    save_splits = not args.no_splits

    print("\n" + "="*70)
    print("DATASET PREPROCESSING PIPELINE")
    print("="*70)
    print(f"Input file:           {input_file}")
    print(f"Output file:          {output_file}")
    print(f"Regressors:           {param_cols}")
    print(f"Outcome variable (Y): {outcome_col}")
    print(f"Scaling method:       {scaling_method.upper()}")
    print("="*70)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_file)
    df = step1_remove_duplicates(df, param_cols)
    df = step2_missing_values(df)
    step3_zero_variance(df, param_cols)
    step4_outcome_distribution(df, outcome_col, save_plots)
    df = step5_handle_outliers(df, outcome_col)
    step6_multicollinearity(df, param_cols, save_plots)

    # Split before scaling to avoid data leakage
    df_train, df_test = step7_train_test_split(df, param_cols, outcome_col)
    df_train, df_test, scaler = step8_feature_scaling(df_train, df_test, param_cols, scaling_method)

    # Coverage check on original unscaled data
    step9_coverage_check(df, param_cols)

    # Save outputs
    print(f"\n{'='*70}")
    print("SAVING OUTPUTS")
    print(f"{'='*70}")

    # Apply the train-fitted scaler to the full dataset for cleaned_dataset.csv
    df_full_scaled = df.copy()
    df_full_scaled[param_cols] = scaler.transform(df[param_cols])
    df_full_scaled.to_csv(output_file, sep=';', index=False)
    print(f"✓ Cleaned dataset saved to: {output_file} ({len(df_full_scaled)} rows)")

    if save_splits:
        train_path = CLEANED_DIR / 'train_set.csv'
        test_path = CLEANED_DIR / 'test_set.csv'
        df_train.to_csv(train_path, sep=';', index=False)
        df_test.to_csv(test_path, sep=';', index=False)
        print(f"✓ Training set saved to:    {train_path} ({len(df_train)} rows)")
        print(f"✓ Test set saved to:        {test_path} ({len(df_test)} rows)")

    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"✓ {output_file}")
    if save_splits:
        print(f"✓ {CLEANED_DIR / 'train_set.csv'} - Training set (80%)")
        print(f"✓ {CLEANED_DIR / 'test_set.csv'} - Test set (20%)")
    print(f"✓ {CLEANED_DIR / f'scaler_{scaling_method}.pkl'} - Scaler (fit on train only)")
    if save_plots:
        print(f"✓ {ANALYSIS_DIR / f'{outcome_col.lower()}_distribution.png'}")
        print(f"✓ {ANALYSIS_DIR / 'correlation_matrix.png'}")
    print(f"\nColumns added: 'is_outlier', '{outcome_col}_is_zero'")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
