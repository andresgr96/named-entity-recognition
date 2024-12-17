from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from gensim.models import KeyedVectors
import optuna
import numpy as np
import argparse
import os

from utils import *



def create_classifier(train_features, train_targets, model_type='NB', max_iter=1000, use_embeddings=False, minibatch_size=10000):
    """
    Train a machine learning model (logreg, NB, SVM) for NER tags using minibatch learning.

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
    # Handle embeddings: Separate embeddings if required
    if use_embeddings:
        embeddings = np.array([feature['embedding'] for feature in train_features])
        features_no_embeddings = remove_embedding_from_features(train_features)
    else:
        embeddings = None
        features_no_embeddings = train_features

    # Vectorize categorical features
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
        
        # Train with minibatches
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

def create_classifier_with_optuna(train_data, train_targets, max_trials=20):
    """
    Perform hyperparameter tuning for LinearSVC using Optuna with k-fold cross-validation.
    """

    vec = DictVectorizer()
    
    # Handle train embeddings
    train_embeddings = np.array([feature['embedding'] for feature in train_data])
    train_features_no_embeddings = remove_embedding_from_features(train_data)
    train_categorical_features = vec.fit_transform(train_features_no_embeddings)
    train_features_combined = np.hstack([train_categorical_features.toarray(), train_embeddings])

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_features_combined, train_targets), n_trials=max_trials)
    best_params = study.best_params
    print(f"Best Parameters: {best_params}")

    best_model = LinearSVC(**best_params, max_iter=10000, dual=False)
    best_model.fit(train_features_combined, train_targets)

    return best_model, vec, best_params

def objective(trial, features, targets):
    """Objective function for Optuna with k-fold cross-validation."""
    # Hyperparameter space
    C = trial.suggest_float('C', 1e-2, 1, log = True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    loss = trial.suggest_categorical('loss', ['hinge', 'squared_hinge'])
    tol = trial.suggest_float('tol', 1e-4, 1e-1, log = True)

    if (penalty == 'l1' and loss != 'squared_hinge') or (penalty == 'l2' and loss == 'hinge'):
        raise optuna.exceptions.TrialPruned()

    model = LinearSVC(C=C, penalty=penalty, loss=loss, dual=False, max_iter=1000, tol=tol)
    targets = np.array(targets)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []

    for train_index, val_index in skf.split(features, targets):
        X_train, X_val = features[train_index], features[val_index]
        y_train, y_val = targets[train_index], targets[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred, average='weighted')
        f1_scores.append(f1)

    return np.mean(f1_scores)


def classify_data(model, vec, inputdata, outputfile, feature_list, embeddings, features_to_remove, word_embedding_model):
    """
    Classifies data using specified features and writes results to a file.

    Args:
    - model: Trained machine learning model.
    - vec: DictVectorizer for transforming features.
    - inputdata: Path to the input data file.
    - outputfile: Path to save the predicted output.
    - feature_list: List of feature keys to include.

    Writes:
    - A file with predicted labels appended to the input features.
    """


    features, _ = extract_features_and_labels(inputdata, word_embedding_model=word_embedding_model)


    if embeddings:
        features = remove_specified_features(features=features, features_to_remove=features_to_remove)
        embeddings_feats = np.array([feature['embedding'] for feature in features])

    else:
        embeddings_feats = None

    features_no_embeddings = remove_embedding_from_features(features)
    filtered_features = [{k: v for k, v in feature.items() if k in feature_list} for feature in features_no_embeddings]

    print(f"Features used: {filtered_features[0]} + Embeddings: {embeddings_feats}")  # Sanity check
    categorical_features = vec.transform(filtered_features)

    if embeddings:
        features_combined = np.hstack([categorical_features.toarray(), embeddings_feats])
    else:
        features_combined = categorical_features

    predictions = model.predict(features_combined)
    with open(outputfile, 'w') as outfile:
        counter = 0
        for line in open(inputdata, 'r'):
            components = line.rstrip('\n').split()
            if components:
                # Write token and predicted label, excluding the original true label
                token, pos_tag, chunk_tag = components[:3]
                outfile.write(f"{token} {pos_tag} {chunk_tag} {predictions[counter]}\n")
                counter += 1


def main(args):
    os.makedirs(args.results_path, exist_ok=True)
    pred_file_base = os.path.join(args.results_path, "predictions")

    # Load word embeddings
    print("Loading word embeddings...")
    word_embedding_model = KeyedVectors.load_word2vec_format(args.embeddings_file, binary=True)

    # Extract features and targets
    print("Extracting features and labels...")
    train_data, train_targets = extract_features_and_labels(args.train_file, word_embedding_model)
    features_to_remove = ['token', 'capitalized', 'token_length', 'word_frequency', 'token_length_bin', 
                          'prefix_2', 'suffix_2', 'prefix_3', 'suffix_3']
    

    # Normal loop to get results for 3 models
    if args.mode == "default":
        feature_list = [
            'token', 'pos_tag', 'chunk_tag', 'capitalized', 'contains_digit', 
            'word_frequency', 'token_length_bin', 'prev_pos_tag', 'next_pos_tag', 
            'prefix_2', 'suffix_2', 'prefix_3', 'suffix_3', 'embedding'
        ]
        train_data_no_embeddings = remove_embedding_from_features(train_data)
        train_data_svm = remove_specified_features(train_data, features_to_remove)
        features_no_embeddings = feature_list[:-1]  # Exclude embeddings for non-SVM models

        print("Training and evaluating models...")
        for model_name in ['SVM', 'logreg', 'NB']:
            print(f"Training and evaluating model: {model_name}")
            pred_file = f"{pred_file_base}_{model_name}.conll"

            if model_name == "SVM":
                model, vec = create_classifier(
                    train_features=train_data_svm, train_targets=train_targets, 
                    model_type=model_name, max_iter=10000, use_embeddings=True
                )
                feature_list_used = feature_list
            else:
                model, vec = create_classifier(
                    train_features=train_data_no_embeddings, train_targets=train_targets, 
                    model_type=model_name, max_iter=10000, use_embeddings=False
                )
                feature_list_used = features_no_embeddings

            # Classify 
            classify_data(
                model=model, vec=vec, inputdata=args.test_file, outputfile=pred_file,
                feature_list=feature_list_used, embeddings=model_name == "SVM", 
                features_to_remove=features_to_remove, word_embedding_model=word_embedding_model
            )

            print(f"Evaluation results for {model_name}:")
            if args.evaluation == "token":
                evaluate_ner(args.test_file, pred_file)
            else:
                span_based_evaluation(args.test_file, pred_file)

    # Loop for tuning SVM
    elif args.mode == "hyperparam":
        print("Performing hyperparameter tuning for LinearSVC...")
        train_features_svm = remove_specified_features(train_data, features_to_remove)
        
        # Tuning happens here
        best_svm_model, vec, best_params = create_classifier_with_optuna(
            train_features_svm, train_targets, max_trials=20
        )
        print(f"Best Hyperparameters: {best_params}")

        # Classify
        pred_file_test = os.path.join(args.results_path, "svm_test_results.conll")
        classify_data(
            model=best_svm_model, vec=vec, inputdata=args.test_file, outputfile=pred_file_test,
            feature_list=feature_list, embeddings=True, features_to_remove=features_to_remove,
            word_embedding_model=word_embedding_model
        )

        print("Final evaluation results for LinearSVC:")
        if args.evaluation == "token":
            evaluate_ner(args.test_file, pred_file_test)
        else:
            span_based_evaluation(args.test_file, pred_file_test)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="NER Training and Evaluation")
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
        "--evaluation", 
        choices=["token", "span"], 
        default="token", 
        help="Evaluation type: token or span"
    )
    parser.add_argument(
        "--mode", 
        choices=["default", "hyperparam"], 
        default="default", 
        help="Mode of operation: train 3 models or hyperparameter tuning"
    )
    args = parser.parse_args()

    main(args)
