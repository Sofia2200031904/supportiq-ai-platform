import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import HfApi
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export(folder):
    model = AutoModelForSequenceClassification.from_pretrained(folder).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(folder)
    batch = tokenizer(["My account was charged a fee", "I disputed my credit report"],
                      padding=True, return_tensors="pt")

    class Logits(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper = Logits()
    wrapper.model = model
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        shape = {0: "batch", 1: "tokens"}
        with torch.no_grad():
            torch.onnx.export(wrapper, (batch.input_ids, batch.attention_mask), path,
                              input_names=["input_ids", "attention_mask"], output_names=["logits"],
                              dynamic_axes={"input_ids": shape, "attention_mask": shape, "logits": {0: "batch"}},
                              opset_version=17, dynamo=False)
        dest = folder / "model.int8.onnx"
        quantize_dynamic(path, dest, weight_type=QuantType.QInt8)
        session = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        logits = session.run(None, {key: batch[key].numpy() for key in ("input_ids", "attention_mask")})[0]
        if logits.shape != (2, model.config.num_labels) or not np.isfinite(logits).all():
            raise ValueError("Quantized export failed its inference check")
        print(f"ONNX bytes: {path.stat().st_size}; INT8 bytes: {dest.stat().st_size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not args.publish:
        export(args.folder)
        return
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = f"{api.whoami()['name']}/supportiq-distilbert"
    api.create_repo(repo, private=False, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=args.folder,
                      allow_patterns=["config.json", "model.safetensors", "model.int8.onnx",
                                      "tokenizer*.json", "special_tokens_map.json", "vocab.txt"])
    print(f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
