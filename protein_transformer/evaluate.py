import datetime
import torch
import typer
import json
import functools
import pandas as pd
import numpy as np

from pathlib import Path
from typing_extensions import Annotated
from model import AntibodyClassifier
from data import Tokenizer, load_data, BCRDataset, collate_fn
from utils import get_device
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


# create a typer app
app = typer.Typer()


class AntibodyPredictor:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.model.to(device)

    def __call__(self, test_dl):
        y_true, y_pred, y_prob = [], [], []
        with torch.inference_mode():
            for batch in test_dl:
                logits = self.model(batch)

                # truth, and predictions for each batch
                y_true.append(batch["label"].cpu().numpy())
                y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
                y_prob.append(torch.softmax(logits, dim=-1).cpu().numpy())

        return (
            np.concatenate(y_true, axis=None),
            np.concatenate(y_pred, axis=None),
            np.vstack(y_prob),
        )

    @classmethod
    def from_run_id(cls, run_id, divice):
        run_path = Path(f"runs/{run_id}")
        with open(run_path / "args.json", "r") as f:
            kwargs = json.load(f)
        model = AntibodyClassifier(**kwargs)
        best_model_path = run_path / "best_model.pt"
        model.load_state_dict(torch.load(best_model_path))
        return cls(model=model, device=divice)


@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option(help="Name for the training run ID")],
    dataset_loc: Annotated[
        str, typer.Option(help="Path to the test dataset in parquet format")
    ],
    batch_size: Annotated[int, typer.Option(help="Number of samples per batch")] = 64,
) -> dict:

    # load test data
    df = load_data(dataset_loc)
    df.reset_index(inplace=True, drop=True)
    test_ds = BCRDataset(df)

    tokenizer = Tokenizer()
    device = get_device()

    # test dataloader
    collate_fn_partial = functools.partial(
        collate_fn, tokenizer=tokenizer, device=device
    )
    test_dl = DataLoader(test_ds, collate_fn=collate_fn_partial, batch_size=batch_size)

    # load model
    predictor = AntibodyPredictor.from_run_id(run_id, device)

    y_true, y_pred, y_prob = predictor(test_dl)

    # Evaluation metrics
    metrics = {
        "run_id": run_id,
    }

    # auc score
    if y_prob.shape[1] == 2:
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
        metrics["auc_score"] = auc_score

    metrics["precision"], metrics["recall"], metrics["f1"], _ = (
        precision_recall_fscore_support(y_true, y_pred, average="weighted")
    )

    save_path = Path(f"runs/{run_id}")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # Save evaluation metrics
    with open(f"runs/{run_id}/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    app()
