# Named Entity Recognition - Andres Garcia

This repository contains the final codebase for the named entitiy recognition project for the CoNLL-2003 task, carried out for the Machine Learning for NLP master class at VU Amsterdam.

The task is to predict the type of entity such as given a token, such as "Person/PER" and "Organization/ORG" as well as the position of the token within an entity, where "B-" marks the beggining of an entity "B-ORG" and "I-" marks a token inside an entity "I-ORG", finally the label "O" is used for non-entity tokens such as "the". For the complete task and dataset description follow the link below.

Original Paper: https://arxiv.org/abs/cs/0306050

If you are interest in the procedure and results, feel free to read the final report submission og the class found in the link below.

Project Paper: https://drive.google.com/file/d/1K4-1u5SXJgRQ5x6O6O1Y1tek6qK57YAk/view?usp=sharing

## Downloading and Installing Dependencies

1. Download .zip file (teachers) or clone the repository (users)
2. Navigate to the projects main directory
3. Then follow the steps below in the terminal to install dependencies
4. Commands are in Linux CDI but should be very similar in other OS's

```
# If you dont have virtualenv installed run:

python3 -m pip install virtualenv

# Create and activate an environment:

python3 -m virtualenv .venv
source .venv/bin/activate

# Install the dependencies:

pip install -r requirements.txt
```

## Downloading embeddings

Since the google embeddings file is heavy, it wont be provided. The repository below allows you to use git to download them as well as the mirror link:

https://github.com/mmihaltz/word2vec-GoogleNews-vectors.git

Download them and follow the instructions, then place the "GoogleNews-vectors-negative300.bin.gz" file under data/vecs.


## Running the Entity Classification
To run the main code use the main.py script. The argument parser setup allows for you to run the file directly on your editor with the default arguments. To run from terminal:

```
python3 main.py [-h] [--evaluation {token,span}] [--mode {default,hyperparam}] [train_file] [dev_file] [test_file] [results_path] [embeddings_file]
```

#### **Positional Arguments**
| Argument          | Description                          |
|-------------------|--------------------------------------|
| `train_file`      | Path to the training file            |
| `dev_file`        | Path to the development file         |
| `test_file`       | Path to the test file                |
| `results_path`    | Directory to save results            |
| `embeddings_file` | Path to the Word2Vec embeddings file |

#### **Options**
| Option                         | Description                                                    |
|--------------------------------|----------------------------------------------------------------|
| `-h, --help`                   | Show this help message and exit                                |
| `--evaluation {token,span}`    | Evaluation type: `token` or `span`                            |
| `--mode {default,hyperparam}`  | Mode of operation: train 3 models (`default`) or hyperparameter tuning (`hyperparam`) |

## Running Feature Ablation Analysis
To run the feature ablation analysis, you can use the feature_ablation.py script provided. 

### Example Usage
```
python3 feature_ablation.py [-h] [--model {SVM,logreg,NB}] [train_file] [dev_file] [test_file] [results_path] [embeddings_file]
```

### **Positional Arguments**
| Argument          | Description                          |
|-------------------|--------------------------------------|
| `train_file`      | Path to the training file            |
| `dev_file`        | Path to the development file         |
| `test_file`       | Path to the test file                |
| `results_path`    | Directory to save results            |
| `embeddings_file` | Path to the Word2Vec embeddings file |

### **Options**
| Option                  | Description                                                   |
|-------------------------|---------------------------------------------------------------|
| `-h, --help`            | Show this help message and exit                               |
| `--model {SVM,logreg,NB}` | Specify the model type for the feature ablation analysis. Options include `SVM`, `logreg`, or `NB`. |

## Running Error Analysis
To perform error analysis on the NER system, you can use the error_anal.py script provided. 

### Example Usage
To run the error analysis script:

```
python3 error_anal.py [-h] [train_file] [test_file] [results_path] [embeddings_file]
```

### **Positional Arguments**
| Argument          | Description                          |
|-------------------|--------------------------------------|
| `train_file`      | Path to the training file            |
| `test_file`       | Path to the test file                |
| `results_path`    | Directory to save results            |
| `embeddings_file` | Path to the Word2Vec embeddings file |

### **Options**
| Option       | Description                              |
|--------------|------------------------------------------|
| `-h, --help` | Show this help message and exit          |

## Running Error Visualization
To visualize errors from the NER system, use the provided Python script. Navigate to the appropriate directory and execute the script as follows:

```
python3 error_vis.py [-h] [--focus_classes FOCUS_CLASSES [FOCUS_CLASSES ...]] [error_file] [output_dir]

```


### **Positional Arguments**
| Argument      | Description                                     |
|---------------|-------------------------------------------------|
| `error_file`  | Path to the prediction analysis CSV file        |
| `output_dir`  | Directory to save results                      |

### **Options**
| Option                                 | Description                                                                |
|----------------------------------------|----------------------------------------------------------------------------|
| `-h, --help`                           | Show this help message and exit                                            |
| `--focus_classes FOCUS_CLASSES [FOCUS_CLASSES ...]` | List of classes to focus on for error analysis (default: `['I-LOC', 'I-MISC', 'I-ORG']`) |




