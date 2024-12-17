from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def extract_features_and_labels(trainingfile, word_embedding_model, word_counts=None, ngram_range=(2, 3)):
    """
    Extract features for NER, including word frequency, n-grams, and word embeddings.
    """
    # Compute word counts if not provided for BoW
    if word_counts is None:
        word_counts = Counter()
        with open(trainingfile, 'r', encoding='utf8') as infile:
            for line in infile:
                components = line.rstrip('\n').split()
                if components: 
                    token = components[0]
                    word_counts[token] += 1

    data = []
    targets = []
    length_bins = [(1, 3), (4, 6), (7, float('inf'))]
    embedding_dim = word_embedding_model.vector_size  # Embedding dimension

    with open(trainingfile, 'r', encoding='utf8') as infile:
        lines = infile.readlines()

    for i, line in enumerate(lines):
        components = line.rstrip('\n').split()
        if components:  
            token = components[0]
            pos_tag = components[1]
            chunk_tag = components[2]
            ner_tag = components[3]

            # Base features
            feature_dict = {
                'token': token,
                'pos_tag': pos_tag,
                'chunk_tag': chunk_tag,
                'capitalized': token[0].isupper(),
                'token_length': len(token),
                'contains_digit': any(char.isdigit() for char in token),
                'word_frequency': word_counts[token] / len(word_counts)
            }

            # Token length binning
            for idx, (low, high) in enumerate(length_bins):
                if low <= len(token) <= high:
                    feature_dict['token_length_bin'] = f'bin_{idx}'
                    break

            # Add contextual features
            if i > 0 and lines[i - 1].strip():
                feature_dict['prev_pos_tag'] = lines[i - 1].split()[1]
            else:
                feature_dict['prev_pos_tag'] = '<START>'
            if i < len(lines) - 1 and lines[i + 1].strip():
                feature_dict['next_pos_tag'] = lines[i + 1].split()[1]
            else:
                feature_dict['next_pos_tag'] = '<END>'

            # Add n-gram features
            for n in range(ngram_range[0], ngram_range[1] + 1):
                feature_dict[f'prefix_{n}'] = token[:n] if len(token) >= n else '<PAD>'
                feature_dict[f'suffix_{n}'] = token[-n:] if len(token) >= n else '<PAD>'

            # Add word embedding as a dense feature
            if token in word_embedding_model:
                embedding = word_embedding_model[token]
            else:
                embedding = np.zeros(embedding_dim)  # Use a zero vector if token is not in the embeddings
            feature_dict['embedding'] = embedding

            data.append(feature_dict)
            targets.append(ner_tag)

    return data, targets


def read_labels(file_path):
    """
    Helper function to part a CONLL file and extract only the labels,
    """ 
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            components = line.strip().split()
            if components:  
                labels.append(components[-1])  
    return labels


def evaluate_ner(gt_file, pred_file):
    """
    Function to compare the true and predicted labels of two CONLL formatted files and calculate model metrics.
    """ 
    gt_labels = read_labels(gt_file)
    pred_labels = read_labels(pred_file)

    if len(gt_labels) != len(pred_labels):
        raise ValueError("Ground truth and prediction files must have the same number of labeled tokens.")

    labels = sorted(set(gt_labels + pred_labels)) 
    cm = confusion_matrix(gt_labels, pred_labels, labels=labels)

    precision = precision_score(gt_labels, pred_labels, labels=labels, average='weighted', zero_division=0)
    recall = recall_score(gt_labels, pred_labels, labels=labels, average='weighted', zero_division=0)
    f1 = f1_score(gt_labels, pred_labels, labels=labels, average='weighted', zero_division=0)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def extract_spans_from_file(file_path):
    """
    Extract spans from a BIO-labeled file, considering sentence boundaries.
    Each span is represented as (start_index, end_index, label).
    """
    spans = []
    start = None
    current_label = None
    current_index = 0  # Token index within the file

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Empty line denotes a sentence boundary
                if current_label is not None:
                    spans.append((start, current_index - 1, current_label))  
                start = None
                current_label = None
                continue

            token, pos, chunk, label = line.split()

            if label.startswith("B-"):
                if current_label is not None:
                    spans.append((start, current_index - 1, current_label))  
                start = current_index
                current_label = label[2:]
            elif label.startswith("I-") and current_label == label[2:]:
                # Continue current span
                pass
            else:
                if current_label is not None:
                    spans.append((start, current_index - 1, current_label))  
                current_label = None
                start = None

            current_index += 1

    if current_label is not None:  
        spans.append((start, current_index - 1, current_label))

    return spans


def spans_overlap(span1, span2):
    """
    Check if two spans overlap and have the same label.
    """
    start1, end1, label1 = span1
    start2, end2, label2 = span2
    return label1 == label2 and not (end1 < start2 or end2 < start1)


def span_based_evaluation(gt_file, pred_file, check_spans=False):
    """
    Evaluate precision, recall, F1-score, and plot a confusion matrix at the span level.
    Considers sentence boundaries in the span extraction.
    """
    gt_spans = extract_spans_from_file(gt_file)
    pred_spans = extract_spans_from_file(pred_file)

    label_counts = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    # Compute true positives, false negatives, and false positives
    for gt_span in gt_spans:
        matched = False
        for pred_span in pred_spans:
            if spans_overlap(gt_span, pred_span):
                label_counts[gt_span[2]]["TP"] += 1
                confusion[gt_span[2]][pred_span[2]] += 1
                matched = True
                break
        if not matched:
            label_counts[gt_span[2]]["FN"] += 1
            confusion[gt_span[2]]["O"] += 1  

    for pred_span in pred_spans:
        if not any(spans_overlap(gt_span, pred_span) for gt_span in gt_spans):
            label_counts[pred_span[2]]["FP"] += 1
            confusion["O"][pred_span[2]] += 1  

    # Calculate precision, recall, F1-score for each label
    all_labels = sorted(set(label for _, _, label in gt_spans + pred_spans))
    precisions, recalls, f1_scores = {}, {}, {}

    for label in all_labels:
        tp = label_counts[label]["TP"]
        fp = label_counts[label]["FP"]
        fn = label_counts[label]["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions[label] = precision
        recalls[label] = recall
        f1_scores[label] = f1

        print(f"{label} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    # Generate confusion matrix, its too weird so ill leave it out for now.
    matrix = []
    for true_label in all_labels:
        row = [confusion[true_label][pred_label] for pred_label in all_labels]
        matrix.append(row)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Span-Level Confusion Matrix")
    # plt.show()

    # Calculate overall metrics
    overall_tp = sum(counts["TP"] for counts in label_counts.values())
    overall_fp = sum(counts["FP"] for counts in label_counts.values())
    overall_fn = sum(counts["FN"] for counts in label_counts.values())

    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print(f"\nOverall - Precision: {overall_precision:.2f}, Recall: {overall_recall:.2f}, F1-Score: {overall_f1:.2f}")

    if check_spans:
        print(f"GT Spans: {list(gt_spans)[:10]}")
        print(f"Pred Spans: {list(pred_spans)[:10]}")



def remove_embedding_from_features(features):
    """
    Removes the 'embedding' field from a list of feature dictionaries.

    Parameters:
    - features: List of dictionaries containing features.

    Returns:
    - features_no_embeddings: List of feature dictionaries without the 'embedding' field.
    """
    features_no_embeddings = []
    for feature in features:
        feature_copy = feature.copy()  # Avoid modifying the original dictionary
        feature_copy.pop('embedding', None)  # Remove embedding if it exists
        features_no_embeddings.append(feature_copy)
    return features_no_embeddings


def remove_specified_features(features, features_to_remove):
    """
    Removes specified fields from a list of feature dictionaries.

    Parameters:
    - features: List of dictionaries containing features.
    - features_to_remove: List of feature keys to remove.

    Returns:
    - features_pruned: List of feature dictionaries with specified fields removed.
    """
    features_pruned = []
    for feature in features:
        feature_copy = feature.copy()  
        for key in features_to_remove:
            feature_copy.pop(key, None) 
        features_pruned.append(feature_copy)
    return features_pruned

