from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import torch
import os

device = 0 if torch.cuda.is_available() else -1

# parse command line arguments
parser = argparse.ArgumentParser(prog="classify.py", description="Predict labels and evaluate for given test data with gold labels.")
parser.add_argument("--baseline", action="store_true", help="Calculate majority and uniform random baseline.")
parser.add_argument("--model", type=str, help="Path to the model directory.")
parser.add_argument("--eval_data", type=str, help="Path to the test data file.", required=True)
parser.add_argument("--num_labels", type=int, default=6, help="Number of labels")
args = parser.parse_args()

# initialise used label set
if args.num_labels == 2:
    labels = ["1", "2"]

elif args.num_labels == 4:
    labels = ["1", "2", "3", "4"]

elif args.num_labels == 5:
    labels = ["1", "2", "3", "4", "5"]

elif args.num_labels == 6:
    labels = ["1", "2", "3", "4", "5", "6"]

else:
    raise ValueError("Unsupported number of labels.")

lbl2idx = {lbl: idx for idx, lbl in enumerate(labels)}
idx2lbl = {idx: lbl for idx, lbl in enumerate(labels)}

############################
# load and prepare test data
############################
print("\nLoading and preparing test data...\n")

def prepare_data(data_path, separator=" <sep> "):
    """Load and reformat the test data into the same format as test data.
    
    @params:
        data_path: path to the test data file
        separator: separator used to distinguish between question and answer, same as in the training data

    @returns:
        texts: list of texts (question + answer) in the same format as training data
        labels: list of labels
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = [line.strip().split("\t") for line in f.readlines()][1:] # skip header

    labels = [line[-1] for line in data]
    texts = [separator.join(line[:-1]) for line in data]

    return texts, labels

test_data_name = os.path.split(args.eval_data)[-1][:-4]
test_texts, gold_labels = prepare_data(args.eval_data)
print("Gold label distribution: ", sorted(list(Counter(gold_labels).items())))


###############################################
# load model and initialise classifier pipeline
###############################################
print("\nLoading model and pipeline...\n")
if args.model:
    model_name = os.path.split(args.model)[-1]
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels), id2label=idx2lbl, label2id=lbl2idx)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)


######################
# generate predictions
######################
def predict(texts, classifier):
    """Generate predictions for the given texts using the classifier pipeline.
    Saves the predictions and input data to a TSV file.
    
    @params:
        texts: list of texts to classify
        classifier: text classification pipeline

    @returns pred_labels: list of predicted labels
    """
    predictions = classifier(texts)
    pred_labels = [pred['label'] for pred in predictions]
    
    with open(f"predictions/{model_name}_{test_data_name}_predictions.tsv", "w", encoding="utf-8") as f:
        for text, pred in zip(texts, pred_labels):
            question, answer = text.split(" <sep> ")
            f.write(f"{question}\t{answer}\t{pred}\n")
        print(f"Predictions saved to {f'{model_name}_{test_data_name}_predictions.tsv'}")

    return pred_labels


#################################################
# generate evaluation report and confusion matrix
#################################################
def plot_cm(gold_labels, pred_labels, labels):
    """Plot confusion matrix for gold labels and predicted labels.
    
    @params:
        gold_labels: list of gold labels
        pred_labels: list of predicted labels
        labels: list of all possible labels

    @returns None

    @sources:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
    """
    cm_golds = confusion_matrix(gold_labels, gold_labels, labels=labels)
    cm_preds = confusion_matrix(gold_labels, pred_labels, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    disp_golds = ConfusionMatrixDisplay(confusion_matrix=cm_golds, display_labels=labels)
    disp_golds.plot(ax=axes[0], cmap=plt.cm.Blues, colorbar=False)
    axes[0].set_title("Gold Label Distribution")
    axes[0].set_xlabel("True label")

    disp_preds = ConfusionMatrixDisplay(confusion_matrix=cm_preds, display_labels=labels)
    disp_preds.plot(ax=axes[1], cmap=plt.cm.Blues, colorbar=False)
    axes[1].set_title("Predicted Label Distribution")
    
    plt.tight_layout()
    # plt.show()
    cm_path = f"predictions/{model_name}_{test_data_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")


def compute_metrics(gold_labels, pred_labels, labels):
    """Compute and print classification report for the given gold labels and predicted labels.
    
    @params:
        gold_labels: list of gold labels
        pred_labels: list of predicted labels
        labels: list of all possible labels

    @returns report: classification report as a dictionary

    @sources:
        https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.classification_report.html
    """
    report = classification_report(gold_labels, pred_labels, labels=labels, target_names=labels, zero_division=0.0, output_dict=True)

    print("Labels\\Metrics\tPrec\tRec\tF1")
    print("---------------------------------------")
    for label, metrics in report.items():
        if label in labels:
            print(f"{label}\t\t{round(metrics['precision'], 6)}\t{round(metrics['recall'], 6)}\t{round(metrics['f1-score'], 6)}")
        if label == "macro avg":
            print("---------------------------------------")
            print(f"Macro avg\t{round(metrics['precision'], 6)}\t{round(metrics['recall'], 6)}\t{round(metrics['f1-score'], 6)}")
        if label == "weighted avg":
            print(f"Weighted avg\t{round(metrics['precision'], 6)}\t{round(metrics['recall'], 6)}\t{round(metrics['f1-score'], 6)}")
    print("---------------------------------------")
    print(f"Accuracy\t{round(report['accuracy'], 6)}")

    return report


def evaluate(gold_labels, labels):
    """Evaluate the model performance by computing metrics and plotting confusion matrix.
    Saves the evaluation report to a txt file.
    
    @params:
        gold_labels: list of gold labels
        labels: list of all possible labels

    @returns None
    """
    print("\nGenerating predictions...\n")
    pred_labels = predict(test_texts, classifier)
    print("Pred label distribution: ", sorted(list(Counter(pred_labels).items())))

    print("\nGenerating evaluation...")
    print("\nEvaluation report:")
    print("------------------")
    report = compute_metrics(gold_labels, pred_labels, labels)
    print("\n")
    plot_cm(gold_labels, pred_labels, labels) # generate and save confusion matrix

    with open(f"predictions/{model_name}_{test_data_name}_eval_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Labels\\Metrics\tPrec\tRec\tF1\n")
        f.write("---------------------------------------\n")
        for label, metrics in report.items():
            if label in labels:
                f.write(f"{label}\t\t{round(metrics['precision'], 6)}\t{round(metrics['recall'], 6)}\t{round(metrics['f1-score'], 6)}\n")
            if label == "macro avg":
                f.write("---------------------------------------\n")
                f.write(f"Macro avg\t{round(metrics['precision'], 6)}\t{round(metrics['recall'], 6)}\t{round(metrics['f1-score'], 6)}\n")
            if label == "weighted avg":
                f.write(f"Weighted avg\t{round(metrics['precision'], 6)}\t{round(metrics['recall'], 6)}\t{round(metrics['f1-score'], 6)}\n")
        f.write("---------------------------------------\n")
        f.write(f"Accuracy\t{round(report['accuracy'], 6)}\n")
    
    print(f"Evaluation report saved to predictions/{f'{model_name}_{test_data_name}_eval_metrics.txt'}")


###########################
# generate baseline scores
###########################
def baseline_metrics(gold_labels, mode):
    """Generate baseline scores using DummyClassifier with the specified strategy.

    @params:
        gold_labels: list of gold labels
        mode: strategy for DummyClassifier: "most_frequent" for majority class baseline or "uniform" for uniform random baseline.

    @returns None

    @sources:
        https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    if mode == "uniform":
        for seed in [0, 1, 42]:
            dummy = DummyClassifier(strategy=mode, random_state=seed)
            dummy_data = [[0]] * len(gold_labels)  # dummy data, as DummyClassifier requires input data
            dummy.fit(dummy_data, gold_labels) 
            pred_labels_dummy = dummy.predict(dummy_data)

            # calculate f1 score and accuracy for dummy classifier
            dummy_f1 = f1_score(gold_labels, pred_labels_dummy, average='macro', zero_division=0.0)
            print(f"F1 score with seed {seed}:", dummy_f1)
            dummy_acc = accuracy_score(gold_labels, pred_labels_dummy)
            print(f"Accuracy with seed {seed}:", dummy_acc)
    
    else:
        dummy = DummyClassifier(strategy=mode, random_state=42)
        dummy_data = [[0]] * len(gold_labels)  # dummy data, as DummyClassifier requires input data
        dummy.fit(dummy_data, gold_labels) 
        pred_labels_dummy = dummy.predict(dummy_data)

        # calculate f1 score and accuracy for dummy classifier
        dummy_f1 = f1_score(gold_labels, pred_labels_dummy, average='macro', zero_division=0.0)
        print("F1 score:", dummy_f1)
        dummy_acc = accuracy_score(gold_labels, pred_labels_dummy)
        print("Accuracy:", dummy_acc)


if __name__ == "__main__":
    # generate baseline scores if requested
    if args.baseline:
        # majority baseline
        print("\nCalculating majority baseline scores...\n")
        baseline_metrics(gold_labels, mode="most_frequent")

        # uniform random baseline
        print("\nCalculating uniform random baseline scores...\n")
        baseline_metrics(gold_labels, mode="uniform")

    else:
        # run evaluation
        evaluate(gold_labels, labels)