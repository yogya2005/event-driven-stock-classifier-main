"""
Master Script: Run Full Pipeline on ALL FNSPID Data

This script runs the complete ML pipeline on the entire FNSPID dataset:
1. Load and filter all data (13+ million rows)
2. Enrich with stock data from yfinance
3. Feature engineering
4. Train models
5. Create visualizations

WARNING: This will take SEVERAL HOURS to complete due to:
- Large dataset size (13+ million rows)
- Many API calls to yfinance (thousands of unique stocks)
- Training models on larger dataset

Estimated time: 4-8 hours depending on system resources and network speed
"""

import subprocess
import sys
import time

def run_task(script_name, task_num, task_name):
    """Run a task script and handle errors."""
    print("\n" + "=" * 80)
    print(f"STARTING TASK {task_num}: {task_name}")
    print("=" * 80)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )

        elapsed = time.time() - start_time
        print(f"\n✓ Task {task_num} completed in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Task {task_num} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False

def main():
    print("=" * 80)
    print("FULL PIPELINE: Event-Driven Stock Impact Classifier")
    print("Processing ALL FNSPID Data")
    print("=" * 80)
    print("\nWARNING: This will take several hours to complete.")
    print("The script will save progress along the way, so you can stop and resume.")
    print("\nPress Ctrl+C at any time to stop (progress will be saved).\n")

    overall_start = time.time()

    # Task 1: Load and filter all data
    if not run_task('task1_load_all_data.py', 1, "Load and Filter ALL FNSPID Data"):
        print("\nPipeline stopped due to Task 1 failure")
        return

    # Task 2: Enrich with stock data (longest task)
    if not run_task('task2_enrich_all_data.py', 2, "Enrich with Stock Data"):
        print("\nPipeline stopped due to Task 2 failure")
        return

    # Task 3: Feature engineering
    if not run_task('task3_feature_engineering_all.py', 3, "Feature Engineering"):
        print("\nPipeline stopped due to Task 3 failure")
        return

    # Task 4: Train models
    if not run_task('task4_train_models_all.py', 4, "Train and Evaluate Models"):
        print("\nPipeline stopped due to Task 4 failure")
        return

    # Task 5: Create visualizations
    if not run_task('task5_create_visualizations_all.py', 5, "Create Visualizations"):
        print("\nPipeline stopped due to Task 5 failure")
        return

    # Pipeline complete
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 80)
    print("✓ FULL PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nTotal time: {overall_elapsed/60:.1f} minutes ({overall_elapsed/3600:.1f} hours)")
    print("\nGenerated files:")
    print("  Data:")
    print("    - ../data/fnspid_all_cleaned.csv")
    print("    - ../data/stock_news_all_labeled.csv")
    print("    - ../data/train_test_split_all.npz")
    print("  Models:")
    print("    - ../models/logistic_regression_model_all.pkl")
    print("    - ../models/random_forest_model_all.pkl")
    print("    - ../models/xgboost_model_all.pkl")
    print("    - ../models/tfidf_vectorizer_all.pkl")
    print("    - ../models/feature_scaler_all.pkl")
    print("    - ../models/model_results_all.pkl")
    print("  Visualizations:")
    print("    - ../visualizations/model_comparison_all.png")
    print("    - ../visualizations/confusion_matrices_all.png")
    print("    - ../visualizations/label_distribution_all.png")
    print("    - ../visualizations/dataset_sample_all.png")
    print("\nYou can now analyze the results and compare with your 2020-only baseline!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        print("Progress has been saved. You can resume by running individual task scripts.")
        sys.exit(0)
