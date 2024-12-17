import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import argparse
from gensim.models import KeyedVectors
import numpy as np
import os
from utils import *
import pandas as pd

def analyze_predictions(test_features, test_targets, predictions, vec, low_classes, output_dir):
    """
    Analyze predictions, saving both correct and incorrect classifications,
    and focusing on specific classes if specified.

    Parameters:
    - test_features: List of feature dictionaries for testing.
    - test_targets: List of true NER tags for testing.
    - predictions: Predicted NER tags from the model.
    - vec: DictVectorizer to inverse transform features.
    - low_classes: Classes to focus on for detailed analysis (e.g., ['I-LOC', 'I-MISC', 'I-ORG']).
    - output_dir: Directory to save analysis results.

    Saves:
    - A CSV file with detailed classification information for all predictions.
    - A CSV file focusing on the specified low-performing classes.
    """
    results = []
    focus_errors = []

    for i, (features, true_label, predicted_label) in enumerate(zip(test_features, test_targets, predictions)):
        # Separate categorical features and embeddings
        features_no_embeddings = remove_embedding_from_features([features])[0]
        
        # Inverse transform only the categorical features
        original_features = vec.inverse_transform(vec.transform([features_no_embeddings]))[0]
        
        is_correct = true_label == predicted_label
        results.append({
            "Token": features["token"],
            "True Label": true_label,
            "Predicted Label": predicted_label,
            "Features": original_features,
            "Correct": is_correct
        })

        if true_label in low_classes and not is_correct:
            focus_errors.append({
                "Token": features["token"],
                "True Label": true_label,
                "Predicted Label": predicted_label,
                "Features": original_features
            })

    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "prediction_analysis.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Prediction analysis saved to {results_file}")

    focus_errors_df = pd.DataFrame(focus_errors)
    if not focus_errors_df.empty:
        focus_errors_file = os.path.join(output_dir, "focus_error_analysis.csv")
        focus_errors_df.to_csv(focus_errors_file, index=False)
        print(f"Focus class error analysis saved to {focus_errors_file}")
    else:
        print("No errors for the specified focus classes.")

    return results_df

def create_svm_classifier(train_features, train_targets, max_iter=10000):
    """
    Train a LinearSVC model for NER tags.

    Parameters:
    - train_features: List of feature dictionaries.
    - train_targets: List of NER tags (target labels).
    - max_iter: Maximum iterations for SVM training.

    Returns:
    - model: Trained LinearSVC model.
    - vec: Fitted DictVectorizer for categorical features.
    """
    embeddings = np.array([feature['embedding'] for feature in train_features])
    features_no_embeddings = remove_embedding_from_features(train_features)
    features_to_remove = ['token', 'capitalized', 'token_length', 'word_frequency', 
                          'token_length_bin', 'prefix_2', 'suffix_2', 'prefix_3', 'suffix_3']
    filtered_features = remove_specified_features(features_no_embeddings, features_to_remove)
    vec = DictVectorizer()
    categorical_features = vec.fit_transform(filtered_features)
    combined_features = np.hstack([categorical_features.toarray(), embeddings])

    model = LinearSVC(max_iter=max_iter)
    model.fit(combined_features, train_targets)
    return model, vec

def evaluate_svm(model, vec, test_features, test_targets, output_dir):
    """
    Evaluate the SVM model, print class-wise and overall metrics, and save a confusion matrix plot.

    Parameters:
    - model: Trained SVM model.
    - vec: Fitted DictVectorizer.
    - test_features: List of feature dictionaries for testing.
    - test_targets: List of true NER tags for testing.
    - output_dir: Directory to save confusion matrix and metrics.
    """
    embeddings = np.array([feature['embedding'] for feature in test_features])
    features_no_embeddings = remove_embedding_from_features(test_features)

    # Remove specific features for SVM and vectorize
    features_to_remove = ['token', 'capitalized', 'token_length', 'word_frequency', 
                          'token_length_bin', 'prefix_2', 'suffix_2', 'prefix_3', 'suffix_3']
    filtered_features = remove_specified_features(features_no_embeddings, features_to_remove)
    categorical_features = vec.transform(filtered_features)

    # Predict and evaluate
    combined_features = np.hstack([categorical_features.toarray(), embeddings])
    predictions = model.predict(combined_features)
    report = classification_report(test_targets, predictions, output_dict=True)
    overall_metrics = precision_recall_fscore_support(test_targets, predictions, average='weighted')

    print("Class-wise Metrics:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"Class {label}: Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
    print("\nOverall Metrics:")
    print(f"Precision: {overall_metrics[0]:.3f}, Recall: {overall_metrics[1]:.3f}, F1: {overall_metrics[2]:.3f}")

    # Save class-wise metrics
    metrics_path = os.path.join(output_dir, "svm_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Class-wise Metrics:\n")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"Class {label}: Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}\n")
        f.write("\nOverall Metrics:\n")
        f.write(f"Precision: {overall_metrics[0]:.3f}, Recall: {overall_metrics[1]:.3f}, F1: {overall_metrics[2]:.3f}\n")

    # Error analysis starts here
    low_classes = ['I-LOC', 'I-MISC', 'I-ORG']  # Focus on these classes
    predictions_df = analyze_predictions(test_features, test_targets, predictions, vec, low_classes, output_dir)

    # Display a few examples
    print("Sample Predictions:")
    print(predictions_df.head())

    # Plot confusion matrix
    cm = confusion_matrix(test_targets, predictions, labels=model.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "error_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")


def main(args):
    os.makedirs(args.results_path, exist_ok=True)

    print("Loading word embeddings...")
    word_embedding_model = KeyedVectors.load_word2vec_format(args.embeddings_file, binary=True)

    print("Extracting features and labels...")
    train_features, train_targets = extract_features_and_labels(args.train_file, word_embedding_model)
    test_features, test_targets = extract_features_and_labels(args.test_file, word_embedding_model)

    print("Training SVM...")
    model, vec = create_svm_classifier(train_features, train_targets)

    print("Evaluating SVM...")
    evaluate_svm(model, vec, test_features, test_targets, args.results_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NER Error Analysis with SVM")
    parser.add_argument(
        "train_file", 
        type=str,
        nargs="?",
        default="./data/conll2003/conll2003.train.conll", 
        help="Path to the training file"
    )
    parser.add_argument(
        "test_file", 
        type=str,
        nargs="?", 
        default="./data/conll2003/conll2003.test.conll", 
        help="Path to the test file"
    )
    parser.add_argument(
        "results_path", 
        type=str,
        nargs="?", 
        default="./results/", 
        help="Directory to save results"
    )
    parser.add_argument(
        "embeddings_file", 
        type=str,
        nargs="?", 
        default="./data/vecs/GoogleNews-vectors-negative300.bin.gz", 
        help="Path to the Word2Vec embeddings file"
    )
    args = parser.parse_args()

    main(args)
