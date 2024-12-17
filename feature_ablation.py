from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.feature_extraction import DictVectorizer
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score
import numpy as np
import argparse
import os

from utils import *


def create_classifier(train_features, train_targets, model_type='SVM', max_iter=1000, use_embeddings=False, minibatch_size=10000):
    """
    Train a machine learning model (logreg, NB, SVM) for NER tags.

    Parameters:
    - train_features: List of feature dictionaries.
    - train_targets: List of NER tags (target labels).
    - model_type: Model type ('logreg', 'NB', 'SVM').
    - max_iter: Maximum iterations for models (used by logreg and SVM).
    - use_embeddings: Whether to include embeddings as features (used for SVM).
    - minibatch_size: Size of minibatches for partial_fit (used for NB).

    Returns:
    - model: Trained model.
    - vec: Fitted DictVectorizer for categorical features.
    """
    if use_embeddings:
        embeddings = np.array([feature['embedding'] for feature in train_features])
        features_no_embeddings = remove_embedding_from_features(train_features)
    else:
        embeddings = None
        features_no_embeddings = train_features

    vec = DictVectorizer()
    categorical_features = vec.fit_transform(features_no_embeddings)

    if use_embeddings:
        features_combined = np.hstack([categorical_features.toarray(), embeddings])
    else:
        features_combined = categorical_features

    if model_type == 'logreg':
        model = LogisticRegression(max_iter=max_iter)
        model.fit(features_combined, train_targets)
    elif model_type == 'NB':
        model = MultinomialNB()
        classes = np.unique(train_targets)
        features_combined, train_targets = shuffle(features_combined, train_targets, random_state=42)
        for start in range(0, len(train_targets), minibatch_size):
            end = start + minibatch_size
            batch_features = features_combined[start:end]
            batch_targets = train_targets[start:end]
            model.partial_fit(batch_features, batch_targets, classes=classes)
    elif model_type == 'SVM':
        model = LinearSVC(max_iter=max_iter)
        model.fit(features_combined, train_targets)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, vec


def feature_ablation_analysis(train_data, train_targets, test_data, test_targets, feature_list, model_type, results_path, use_embeddings, word_embedding_model):
    """
    Perform feature ablation analysis by iteratively removing features and evaluating performance.

    Parameters:
    - train_data: List of feature dictionaries for training.
    - train_targets: List of target labels for training.
    - test_data: List of feature dictionaries for testing.
    - test_targets: List of target labels for testing.
    - feature_list: List of all features.
    - model_type: The type of model to use ('SVM', 'logreg', 'NB').
    - results_path: Path to save results.
    - use_embeddings: Whether to include embeddings.
    - word_embedding_model: Loaded word embedding model.
    """
    os.makedirs(results_path, exist_ok=True)

    # Define features to pre-remove for SVM
    features_to_remove = ['token', 'capitalized', 'token_length', 'word_frequency', 'token_length_bin',
                          'prefix_2', 'suffix_2', 'prefix_3', 'suffix_3']
    
    # Adjust training and test data for SVM
    if model_type == "SVM":
        train_data = remove_specified_features(train_data, features_to_remove)
        test_data = remove_specified_features(test_data, features_to_remove)

    # Separate embeddings and other features for the baseline model
    if use_embeddings:
        train_embeddings = np.array([feature['embedding'] for feature in train_data])
        test_embeddings = np.array([feature['embedding'] for feature in test_data])
    else:
        train_embeddings = None
        test_embeddings = None

    train_features_no_embeddings = remove_embedding_from_features(train_data)
    test_features_no_embeddings = remove_embedding_from_features(test_data)

    vec = DictVectorizer()
    train_categorical_features = vec.fit_transform(train_features_no_embeddings)
    test_categorical_features = vec.transform(test_features_no_embeddings)

    if use_embeddings:
        train_features_combined = np.hstack([train_categorical_features.toarray(), train_embeddings])
        test_features_combined = np.hstack([test_categorical_features.toarray(), test_embeddings])
    else:
        train_features_combined = train_categorical_features
        test_features_combined = test_categorical_features

    print(f"Features used: {train_features_no_embeddings[0]} + Embeddings?: {use_embeddings}")  # Sanity check    

    # Train and evaluate the baseline model
    print(f"Training Baseline Model")
    if model_type == "logreg":
        baseline_model = LogisticRegression(max_iter=1000)
    elif model_type == "NB":
        baseline_model = MultinomialNB()
    elif model_type == "SVM":
        baseline_model = LinearSVC(max_iter=10000)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    baseline_model.fit(train_features_combined, train_targets)
    print("Evaluating Baseline")
    baseline_score = f1_score(test_targets, baseline_model.predict(test_features_combined), average="weighted")
    print(f"Baseline F1-Score (All Features): {baseline_score:.4f}")

    results = {"baseline": baseline_score}

    # Perform ablation for each feature
    for feature in feature_list:
        print(f"Ablating feature: {feature}")
        if feature == "embedding":
            use_embeddings = False
        ablated_train_features = remove_specified_features(train_data, [feature])
        ablated_test_features = remove_specified_features(test_data, [feature])

        ablated_train_no_embeddings = remove_embedding_from_features(ablated_train_features)
        ablated_test_no_embeddings = remove_embedding_from_features(ablated_test_features)

        # Re-vectorize the ablated features
        ablated_train_categorical = vec.transform(ablated_train_no_embeddings)
        ablated_test_categorical = vec.transform(ablated_test_no_embeddings)

        if use_embeddings:
            ablated_train_combined = np.hstack([ablated_train_categorical.toarray(), train_embeddings])
            ablated_test_combined = np.hstack([ablated_test_categorical.toarray(), test_embeddings])
        else:
            ablated_train_combined = ablated_train_categorical
            ablated_test_combined = ablated_test_categorical


        print(f"Features used: {ablated_train_no_embeddings[0]} + Embeddings?: {use_embeddings}")  # Sanity check    

        # Train and evaluate with the ablated features
        if model_type == "logreg":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "NB":
            model = MultinomialNB()
        elif model_type == "SVM":
            model = LinearSVC(max_iter=10000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(ablated_train_combined, train_targets)
        ablated_score = f1_score(test_targets, model.predict(ablated_test_combined), average="weighted")
        print(f"F1-Score without {feature}: {ablated_score:.4f}")
        results[feature] = ablated_score

    # Save results
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    results_path_file = os.path.join(results_path, f"feature_ablation_results_{args.model}.txt")
    with open(results_path_file, "w") as f:
        for feature, score in sorted_results:
            f.write(f"{feature}: {score:.4f}\n")

    print("Feature ablation analysis complete. Results saved.")
    print(f"Results saved to: {results_path_file}")



def main(args):
    os.makedirs(args.results_path, exist_ok=True)

    print("Loading word embeddings...")
    word_embedding_model = KeyedVectors.load_word2vec_format(args.embeddings_file, binary=True)

    print("Extracting features and labels...")
    train_data, train_targets = extract_features_and_labels(args.train_file, word_embedding_model)
    test_data, test_targets = extract_features_and_labels(args.test_file, word_embedding_model)

    feature_list = [
        'token', 'pos_tag', 'chunk_tag', 'capitalized', 'contains_digit',
        'word_frequency', 'token_length_bin', 'prev_pos_tag', 'next_pos_tag',
        'prefix_2', 'suffix_2', 'prefix_3', 'suffix_3', 'embedding'
    ]

    feature_list_svm = [
        'pos_tag', 'chunk_tag', 'contains_digit', 'prev_pos_tag', 'next_pos_tag', 'embedding'
    ]


    print(f"Performing feature ablation analysis on {args.model}...")
    feature_ablation_analysis(
        train_data=train_data,
        train_targets=train_targets,
        test_data=test_data,
        test_targets=test_targets,
        feature_list=feature_list if not (args.model == "SVM") else feature_list_svm,
        model_type=args.model,
        results_path=args.results_path,
        use_embeddings=(args.model == "SVM"),
        word_embedding_model=word_embedding_model,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature Ablation Analysis for NER")
    parser.add_argument(
        "train_file",
        type=str,
        nargs="?",
        default="./data/conll2003/conll2003.train.conll",
        help="Path to the training file"
    )
    parser.add_argument(
        "dev_file",
        type=str,
        nargs="?",
        default="./data/conll2003/conll2003.dev.conll",
        help="Path to the development file"
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
    parser.add_argument(
        "--model",
        type=str,
        choices=["SVM", "logreg", "NB"],
        default="NB",
        help="Model type for feature ablation analysis"
    )
    args = parser.parse_args()

    main(args)
