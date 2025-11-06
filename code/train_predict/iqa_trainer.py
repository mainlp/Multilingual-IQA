from transformers import Trainer
from datasets import Dataset
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption
from typing import Optional, Union
from collections import Counter
import numpy as np
import os
import time
import math

class IQATrainer(Trainer):
    """Create a custom Trainer class to modify the logging and evaluation functions."""
    def __init__(self, *args, meta=None, **kwargs):
        """Inherit the init from the parent class and add own arguments."""
        super().__init__(*args, **kwargs)
        self.curr_train_loss = None
        self.log_history = ["Epoch\tTrain_Loss\tEval_Loss\tAccuracy\tPrecision\tRecall\t\tF1"]
        self.curr_score = [0, 0] # [epoch, accuracy]
        self.curr_best = [0, 0, 0] # [epoch, accuracy, f1]
        self.curr_epoch_predictions = []
        self.curr_best_predictions = []

        # read additional data
        self.log_dir = meta["log_dir"]
        self.model_id_finetuned = meta["model_id_finetuned"]
        self.log_text = meta["log_text"]
        self.labels = meta["labels"]
        self.lbl2idx = meta["lbl2idx"]
        self.test_label_counts = meta["test_label_counts"]
        
    def log(self, logs, start_time=None):
        """Custom logging output."""
        epoch = round(self.state.epoch or 0)

        # print default output from log function for checking during training
        print(logs)
            
        if "loss" in logs.keys(): # train scores
            # save current loss so that it can be printed with the eval scores in the next call
            self.curr_loss = round(logs["loss"], 6)

        if "eval_loss" in logs.keys(): # eval scores
            eval_loss = round(logs["eval_loss"], 6)
            eval_accuracy = round(logs["eval_accuracy"], 6)
            eval_precision = round(logs["eval_precision"], 6)
            eval_recall = round(logs["eval_recall"], 6)
            eval_f1 = round(logs["eval_f1"], 6)

            epoch_results = f"{epoch}\t{self.curr_loss}\t{eval_loss}\t{eval_accuracy}\t{eval_precision}\t{eval_recall}\t{eval_f1}"
            self.curr_score = [epoch, eval_accuracy]

            # add epoch evaluation to history
            self.log_history.append(epoch_results)
            
            # check if it is better than current best
            if eval_accuracy > self.curr_best[1]:
                self.curr_best = [epoch, eval_accuracy, eval_f1]

    def print_full_results(self):
        """Print the full results of the training."""
        print("\n")
        for row in self.log_history:
            print(row)
        print(f"\nBest Epoch is #{self.curr_best[0]} with accuracy score of {self.curr_best[1]} and f1 score of {self.curr_best[2]}")

    def write_full_results(self):
        """Write the full results of the training to a log file."""
        # check if directory exists, if not create it
        os.makedirs(self.log_dir, exist_ok=True)

        epoch = round(self.state.epoch)
        log_path = os.path.join(self.log_dir, f"log-{self.model_id_finetuned}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            # write params to log file
            f.write(self.log_text)

            # print train evaluation scores
            f.write("\n\nTrain evaluation scores:\n")
            for row in self.log_history:
                f.write(row + "\n")
            f.write(f"\nBest Epoch is #{self.curr_best[0]} with accuracy score of {self.curr_best[1]} and f1 score of {self.curr_best[2]}")

            # write predicted label distribution of best epoch
            gold_counts, predicted_counts, _ = self.curr_best_predictions
            f.write(f"\n\nPredicted label distribution in best epoch {self.curr_best[0]}:\n")
            f.write("\tLabel\tGold\tPred\n")
            for label in self.labels:
                f.write(f"\t{label}\t{gold_counts.get(self.lbl2idx[label], 0)}\t{predicted_counts.get(self.lbl2idx[label], 0)}\n")

            # write predicted label distribution of last epoch
            f.write(f"\n\nPredicted label distribution in last epoch {epoch}:\n")
            f.write("\tLabel\tGold\tPred\n")
            for label in self.labels:
                f.write(f"\t{label}\t{self.test_label_counts.get(label, 0)}\t{self.curr_epoch_predictions.get(self.lbl2idx[label], 0)}\n")

            f.write(f"\n*** ***")
            
        print(f"\nLog file saved to {self.log_dir}")

    def _print_predicted_label_dist(self, gold_counts, predicted_counts, epoch):
        """"Custom function to print the predicted label distribution of the current epoch."""
        print(f"\nPredicted label distribution in epoch {epoch}:")
        print("\tLabel\tGold\tPred")
        for label in self.labels:
            print(f"\t{label}\t{gold_counts.get(self.lbl2idx[label], 0)}\t{predicted_counts.get(self.lbl2idx[label], 0)}")
        print("")

    def evaluate(self, eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None, ignore_keys: Optional[list[str]] = None, metric_key_prefix: str = "eval",) -> dict[str, float]:
        """Modification of the original evaluate function.
        Inserts custom function _print_predicted_label_dist() to print the predicted label distribution per epoch in the Custom Code section.
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


        ### Custom Code ###
        # retrieve gold and predicted labels from the model output
        gold_label_counts = Counter(output.label_ids)
        predicted_labels = np.argmax(output.predictions, axis=1)
        predicted_label_counts = Counter(predicted_labels)
        self.curr_epoch_predictions = predicted_label_counts

        # print predicted label distribution of the current epoch
        self._print_predicted_label_dist(gold_label_counts, predicted_label_counts, round(self.state.epoch))

        if self.curr_best[-1] < output.metrics["eval_accuracy"]:
        # save the predicted label distribution of the best epoch
            self.curr_best_predictions = (gold_label_counts, predicted_label_counts, round(self.state.epoch))
        ### Custom Code end ###


        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics