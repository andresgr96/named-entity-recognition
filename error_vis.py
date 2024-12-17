import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def analyze_error_patterns(error_file, output_dir, focus_classes):
    """
    Perform error analysis using the saved CSV file.

    Parameters:
    - error_file: Path to the CSV file containing prediction analysis data.
    - output_dir: Directory to save plots and summaries.
    - focus_classes: List of classes to focus on for the analysis.

    Returns:
    - None (prints and saves outputs).
    """
    df = pd.read_csv(error_file)

    if focus_classes:
        df = df[df['True Label'].isin(focus_classes)]

    # Summarize frequent misclassifications
    misclassification_summary = df.groupby(['True Label', 'Predicted Label']).size().reset_index(name='Count')
    misclassification_summary = misclassification_summary.sort_values(by='Count', ascending=False)
    print("Frequent Misclassifications:")
    print(misclassification_summary)

    os.makedirs(output_dir, exist_ok=True)
    misclassification_summary_path = os.path.join(output_dir, "misclassification_summary.csv")
    misclassification_summary.to_csv(misclassification_summary_path, index=False)
    print(f"Misclassification summary saved to {misclassification_summary_path}")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=misclassification_summary.head(10),
        x='Count',
        y='True Label',
        hue='Predicted Label',
        dodge=False
    )
    plt.title("Top 10 Frequent Misclassifications")
    plt.xlabel("Count")
    plt.ylabel("True Label")
    plt.legend(title="Predicted Label")
    plt.tight_layout()
    frequent_misclassifications_path = os.path.join(output_dir, "frequent_misclassifications.png")
    plt.savefig(frequent_misclassifications_path)
    print(f"Frequent misclassifications plot saved to {frequent_misclassifications_path}")
    plt.show()

    # Feature correlation for common errors
    print("\nAnalyzing Feature Correlations...")
    for true_label, predicted_label in misclassification_summary[['True Label', 'Predicted Label']].values[:3]:
        subset = df[(df['True Label'] == true_label) & (df['Predicted Label'] == predicted_label)]
        print(f"\nError Analysis for True Label: {true_label}, Predicted Label: {predicted_label}")
        feature_counts = pd.Series(subset['Features'].values).value_counts()
        print(f"Frequent Features: \n{feature_counts.head()}")

    # Compare correct vs. incorrect predictions
    print("\nComparing Correct vs. Incorrect Predictions...")
    for label in df['True Label'].unique():
        # Correct predictions
        correct_subset = df[(df['True Label'] == label) & (df['True Label'] == df['Predicted Label'])]
        incorrect_subset = df[(df['True Label'] == label) & (df['True Label'] != df['Predicted Label'])]
        print(f"\nLabel: {label}")
        print(f"Correct Predictions: {len(correct_subset)}")
        print(f"Incorrect Predictions: {len(incorrect_subset)}")

        if not incorrect_subset.empty:
            incorrect_features = pd.Series(incorrect_subset['Features'].values).value_counts()
            print(f"Frequent Incorrect Features for {label}: \n{incorrect_features.head()}")

    # Identify classes with minimal errors
    minimal_error_classes = df.groupby("True Label").apply(
        lambda x: (x['True Label'] == x['Predicted Label']).sum()
    )
    print("\nClasses with Minimal Errors:")
    print(minimal_error_classes.sort_values(ascending=False).head(5))


def main():
    parser = argparse.ArgumentParser(description="NER Error Analysis")
    parser.add_argument(
        "error_file",
        nargs="?",
        default="./results/prediction_analysis.csv", 
        type=str,
        help="Path to the prediction analysis CSV file."
    )
    parser.add_argument(
        "output_dir", 
        type=str,
        nargs="?", 
        default="./results/", 
        help="Directory to save results"
    )
    parser.add_argument(
        "--focus_classes",
        nargs="+",
        default=['I-LOC', 'I-MISC', 'I-ORG'], 
        help="List of classes to focus on for error analysis (default: ['I-LOC', 'I-MISC', 'I-ORG'])"
    )
    args = parser.parse_args()

    analyze_error_patterns(args.error_file, args.output_dir, args.focus_classes)


if __name__ == "__main__":
    main()
