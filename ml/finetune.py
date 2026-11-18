import argparse
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import confusion_matrix
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

from ml.data import load_splits, verify_splits
from ml.evaluate import class_weights, score_predictions

BASE = "distilbert/distilbert-base-uncased"


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=self.weights.to(labels.device))
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Select a GPU runtime in Colab")
    frames, manifest = load_splits(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    labels = sorted(frames["train"].topic.unique())
    ids = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(labels), label2id=ids, id2label=dict(enumerate(labels)))

    def encode(frame):
        data = Dataset.from_dict({"text": frame.text.tolist(), "labels": frame.topic.map(ids).tolist()})
        return data.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=256),
                        batched=True, remove_columns=["text"])

    def metrics(pred):
        scores = score_predictions(pred.label_ids, pred.predictions.argmax(axis=1), list(range(len(labels))))
        return {key: scores[key] for key in ("macro_f1", "accuracy")}

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///tracking.db"))
    mlflow.set_experiment("supportiq-distilbert")
    config = TrainingArguments(
        output_dir=str(args.output / "checkpoints"), per_device_train_batch_size=16,
        per_device_eval_batch_size=32, num_train_epochs=4, learning_rate=2e-5,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, fp16=True, seed=42, report_to=["mlflow"])
    trainer = WeightedTrainer(model=model, args=config, train_dataset=encode(frames["train"]),
                              eval_dataset=encode(frames["val"]), processing_class=tokenizer,
                              data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    weights = class_weights(frames["train"].topic)
    trainer.weights = torch.tensor([weights[label] for label in labels], dtype=torch.float32)
    trainer.model_accepts_loss_kwargs = False
    verify_splits(args.data)
    trainer.train()
    dest = args.output / "model"
    trainer.save_model(str(dest))
    tokenizer.save_pretrained(dest)
    verify_splits(args.data)
    test = encode(frames["test"])
    pred = trainer.predict(test)
    actual = frames["test"].topic
    predicted = np.asarray(labels)[pred.predictions.argmax(axis=1)]
    result = score_predictions(actual, predicted, labels)
    result.update(labels=labels, confusion_matrix=confusion_matrix(actual, predicted, labels=labels).tolist(),
                  split_hashes={key: spec["sha256"] for key, spec in manifest["splits"].items()},
                  checkpoint=trainer.state.best_model_checkpoint, gpu=torch.cuda.get_device_name())
    texts = frames["test"].text.iloc[:1000].tolist()
    model.eval()

    @torch.no_grad()
    def predict_texts():
        for i in range(0, len(texts), 32):
            batch = tokenizer(texts[i:i+32], padding=True, truncation=True, max_length=256, return_tensors="pt")
            model(**batch.to(model.device)).logits.argmax(dim=1)
        torch.cuda.synchronize()

    verify_splits(args.data)
    predict_texts()
    times = []
    for _ in range(3):
        verify_splits(args.data)
        start = time.perf_counter()
        predict_texts()
        times.append((time.perf_counter() - start) * 1000 / len(texts))
    result["seconds_per_1000"] = float(np.median(times))
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with mlflow.start_run(run_name="distilbert-frozen-test"):
        mlflow.log_metrics({key: result[key] for key in ("macro_f1", "accuracy", "seconds_per_1000")})
        mlflow.log_artifact(str(args.output / "results.json"))


if __name__ == "__main__":
    main()
