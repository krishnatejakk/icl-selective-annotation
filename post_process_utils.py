import json
import os
from datasets import load_metric
import nltk


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def post_process_xsum(args, golds, preds):
    assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
    preds, golds = postprocess_text(preds, golds)
    metric = load_metric("rouge")
    result = metric.compute(predictions=preds, references=golds, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    with open(os.path.join(args.output_dir, "result_summary.json"), "w") as f:
        json.dump(result, f)
    return golds, preds
