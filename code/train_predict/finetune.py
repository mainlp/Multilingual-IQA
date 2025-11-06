from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict
from collections import Counter
import numpy as np
import evaluate
import argparse
import shutil
import random
import torch
import os
import json
from iqa_trainer import IQATrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# follows code from https://huggingface.co/docs/transformers/en/tasks/sequence_classification

# read command line arguments
parser = argparse.ArgumentParser(prog="finetune.py", description="Train a model for InQA")
parser.add_argument("--random_search", action="store_true", help="Perform a random search for hyperparameter finetuning")
parser.add_argument("--trials", type=int, default=20, help="Number of trials for random parameter search")
parser.add_argument("--train", action="store_true", help="Train a model with user-defined parameters")
parser.add_argument("--model_name", type=str, default="google-bert/bert-base-multilingual-cased", help="Name of the pre-trained model")
parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
parser.add_argument("--train_path", type=str, default="../../data/IndirectQA/EN/IndirectQA_en_all.tsv", help="Path to the train data file")
parser.add_argument("--eval_path", type=str, default=None, help="Path to the evaluation data file")
parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the model")
parser.add_argument("--log_dir", type=str, default="train_logs", help="Directory to save the logs")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate") # common when finetuning bert
parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size") 
parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay") # common when finetuning bert
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio") # torch default value
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--label_smooth", type=float, default=0.0, help="Label smoothing factor against overfitting")
parser.add_argument("--delete_model", action="store_true", help="Delete the model after training")

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

# generate model output name and path
model_id = args.model_name
dataset_name = os.path.split(args.train_path)[-1][:-4].lower()
model_prefix = model_id.split("/")[-1].split("-")[0]
model_id_finetuned = f"{model_prefix}-{dataset_name}-lr{args.learning_rate:.0e}-s{args.seed}-bs{args.batch_size}-wr{round(args.warmup_ratio, 2)}"
# add more parameters to model_id_finetuned
model_id_finetuned += f"-wd{round(args.weight_decay, 2)}-do{round(args.dropout, 2)}-ls{round(args.label_smooth, 2)}"
output_dir = os.path.join(args.output_dir, dataset_name)
output_path = os.path.join(output_dir, model_id_finetuned)

# check if directory exists, if not create it
os.makedirs(output_dir, exist_ok=True)

log_text = f"""\n*** Training {args.model_name.split("/")[-1]} on {dataset_name} ***
\nParameters:
\tLearning Rate: {args.learning_rate}
\tSeed: {args.seed}
\tBatch Size: {args.batch_size}
\tWeight Decay: {args.weight_decay}
\tWarmup Ratio: {args.warmup_ratio}
\tDropout: {args.dropout}
\tLabel Smoothing Factor: {args.label_smooth}
"""

if args.train:
    print(log_text)


#######################
# open and process data
#######################
print("\nLoad data...")

data = load_dataset("csv", data_files=args.train_path, delimiter="\t", split="train")
question_col, answer_col, label_col = data.column_names

if args.eval_path:
    test_data = load_dataset("csv", data_files=args.eval_path, delimiter="\t", split="train")
    data = DatasetDict({
        "train": data,
        "test": test_data
    })

else:
    data = data.train_test_split(test_size=0.2, shuffle=True, seed=args.seed) 

train_label_counts = sorted(list(Counter(list(row["Annotation (Label)"] for row in data["train"])).items()))
print("\nTrain Total: ", len(data["train"]))
print(f"Train counts: {train_label_counts}")

test_label_counts = Counter(list(row["Annotation (Label)"] for row in data["test"]))
test_label_counts_sorted = sorted(list(test_label_counts.items()))
print("Test Total: ", len(data["test"]))
print(f"Test counts: {test_label_counts_sorted}")

labels = sorted(list(set(row["Annotation (Label)"] for row in data["train"])))
print("Labels: ", labels)

# compute class weights for loss computation as inverse frequency
label_counts = dict(train_label_counts)
total = sum(label_counts.values())
num_classes = len(label_counts)

class_weights = [total / (num_classes * label_counts[lbl]) for lbl in labels]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device) # convert to tensor


lbl2idx = {label: idx for idx, label in enumerate(labels)}
print("Label to index: ", lbl2idx)
idx2lbl = {idx: label for label, idx in lbl2idx.items()}
separator = " [SEP] "

def preprocess_data(sample):
    """Mapping function: Adjust data formatting for model training. Replace string labels with label indexes."""
    text = sample[question_col] + separator + sample[answer_col]
    return {"text": text, "label": lbl2idx[sample[label_col]]}

data = data.map(preprocess_data, remove_columns=["Question", "Answer", "Annotation (Label)"])
data["train"][0]


########################################
# set model arguments and load tokenizer
########################################
print("\nLoad tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(sample):
    """Mapping function: Tokenize the input text."""
    tokenized = tokenizer(sample["text"], truncation=True)
    tokenized["label"] = sample["label"]
    # if "token_type_ids" not in tokenized:
    #     tokenized["token_type_ids"] = None  # omit if model doesn't support token_type_ids
    return tokenized

data_tokenized = data.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


##############
# load metrics
##############
print("\nLoad metrics...")

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def compute_metrics(pred):
    """Compute metrics for the model evaluation.
    
    @params pred: predictions from the model.

    @returns: dictionary with accuracy, precision, recall and f1 score.
    """
    logits, labels = pred
    predictions = np.argmax(logits, axis=1) # get the predicted class with the highest score

    acc_score = accuracy.compute(predictions=predictions, references=labels)
    prec_score = precision.compute(predictions=predictions, references=labels, zero_division=0, average="macro")
    rec_score = recall.compute(predictions=predictions, references=labels, zero_division=0, average="macro")
    f1_score = f1.compute(predictions=predictions, references=labels, average="macro")

    return {"accuracy": acc_score["accuracy"], "precision": prec_score["precision"], "recall": rec_score["recall"], "f1": f1_score["f1"]}


###############################
# load model and set up trainer
###############################
print("\nLoad model and trainer...")

# model ids
# BERT:         google-bert/bert-base-multilingual-cased    https://huggingface.co/google-bert/bert-base-multilingual-cased
# XML-R:        FacebookAI/xlm-roberta-base                 https://huggingface.co/FacebookAI/xlm-roberta-base
# mDeBERTa:     microsoft/mdeberta-v3-base                  https://huggingface.co/microsoft/mdeberta-v3-base

def model_init(trial=None):
    """Create a custom model initialisation function to finetune dropout during hyperparameter search.
    
    @params trial: Is not None during hyperparameter search with trainer.hyperparameter_search.

    @returns loaded model with custom config

    @sources:
        https://huggingface.co/docs/transformers/v4.52.3/en/main_classes/trainer#transformers.Trainer.model_init
        https://huggingface.co/docs/transformers/hpo_train?backends=Ray+Tune
    """
    config_kwargs = {
        "num_labels": len(labels),
        "id2label": idx2lbl,
        "label2id": lbl2idx,
    }

    # set dropout value depending on whether model is searching or training
    # and fit to search method
    if trial:
        dropout = trial.suggest_float("dropout", 0.1, 0.5, log=True) # optuna, random search
    else:
        dropout = args.dropout
    
    config = AutoConfig.from_pretrained(model_id, **config_kwargs)

    # check if model has dropout parameters in its config
    if hasattr(config, "hidden_dropout_prob"):
        config_kwargs["hidden_dropout_prob"] = dropout
    if hasattr(config, "attention_probs_dropout_prob"):
        config_kwargs["attention_probs_dropout_prob"] = dropout

    return AutoModelForSequenceClassification.from_pretrained(model_id, config=config)


training_args = TrainingArguments(
    output_dir=output_path,
    num_train_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    push_to_hub=False,
    gradient_accumulation_steps=1,
    label_smoothing_factor=args.label_smooth,
    weight_decay=args.weight_decay,
    fp16=True,
    warmup_ratio=args.warmup_ratio,
    overwrite_output_dir=True,
    save_only_model=True,
    save_total_limit=1,
    seed=args.seed
)

trainer = IQATrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=data_tokenized["train"],
    eval_dataset=data_tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    meta={ # pass additional data to the custom trainer class init
        "log_dir": args.log_dir,
        "labels": labels,
        "lbl2idx": lbl2idx,
        "log_text": log_text,
        "model_id_finetuned": model_id_finetuned,
        "test_label_counts": test_label_counts,
    }
)

if args.train:
    print("\nStart training...")

    trainer.train(resume_from_checkpoint=False)
    print("\nFinal eval metrics after training:", trainer.evaluate())
    trainer.save_model(output_path)

    trainer.print_full_results()
    trainer.write_full_results()


if args.random_search:
    # follows code from https://huggingface.co/docs/transformers/hpo_train?backends=Ray+Tune
    # dropout is set during the model_initialisation
    print("\nStart hyperparameter search with random search using optuna...")

    def param_space(trial):
        """Parameter search space for random search."""
        params = {
            # choose params from log space so that each has the same chance of being selected
            # https://stats.stackexchange.com/questions/291552/why-do-we-sample-from-log-space-when-optimizing-learning-rate-regularization-p
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
            "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.1, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.1, 0.5, log=True),
            "label_smooth": trial.suggest_float("label_smooth", 0.05, 0.5, log=True)
        }
        print("\nTrial params:", params, "\n")
        return params
    
    def compute_objective(metrics):
        """Return eval_loss at the end of trial."""
        return metrics["eval_loss"]

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        hp_space=param_space,
        n_trials=args.trials,
        backend="optuna",
        compute_objective=compute_objective
    )

    print("\n", best_trial)

    # save best trial params
    best_trial_params = f"trial-best-params-{model_prefix}-{dataset_name}.json"
    best_trial_params_path = os.path.join("configs", best_trial_params)
    with open(best_trial_params_path, "w", encoding="utf-8") as f:
        json.dump(best_trial, f, indent=2)
    print(f"\nSaved best trial parameters to {best_trial_params_path}")

if args.delete_model:
    print(f"\nDeleting model from {output_path}...")
    shutil.rmtree(output_path)

print("\n*** Done! ***\n")