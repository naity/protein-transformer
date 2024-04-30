import torch
import torch.nn as nn
import tempfile
import functools
import ray
import typer
import json

from pathlib import Path
from ray import train, tune
from typing_extensions import Annotated
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from utils import get_device
from model import AntibodyClassifier
from train import get_dataloaders, train_epoch, val_epoch
from data import Tokenizer, load_data

# create a typer app
app = typer.Typer()


def train_loop(config, dataset_loc):
    # Dataset
    df = load_data(dataset_loc)
    tokenizer = Tokenizer()
    device = get_device()

    # train and val dataloaders
    train_dl, val_dl = get_dataloaders(
        df, config["val_size"], tokenizer, device, config["batch_size"]
    )

    model_kwargs = {
        "vocab_size": tokenizer.vocab_size,
        "padding_idx": tokenizer.pad_token_id,
        "embedding_dim": config["embedding_dim"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "ffn_dim": config["embedding_dim"] * 2,
        "dropout": config["dropout"],
        "num_classes": config["num_classes"],
    }

    # model
    model = AntibodyClassifier(**model_kwargs)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    start = 1
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(Path(checkpoint_dir) / "checkpoint.pt")
            start = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])

    model.to(device)

    for epoch in range(start, config["num_epochs"] + 1):
        # Train
        train_loss = train_epoch(model, train_dl, loss_fn, opt)
        # Validation
        val_loss, _, _, _ = val_epoch(model, val_dl, loss_fn)

        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_dict = {
                "model_kwargs": model_kwargs,
                "epoch": epoch,
                "model_state": model.state_dict(),
            }
            torch.save(
                checkpoint_dict,
                Path(tempdir) / "checkpoint.pt",
            )
            train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))


@app.command()
def tune_model(
    run_id: Annotated[str, typer.Option(help="Name for the training run ID")],
    dataset_loc: Annotated[
        str, typer.Option(help="Path to the dataset in parquet format")
    ],
    val_size: Annotated[
        float, typer.Option(help="Proportion of the dataset to use for validation")
    ] = 0.15,
    num_classes: Annotated[
        int, typer.Option(help="Number of final output dimensions")
    ] = 2,
    batch_size: Annotated[int, typer.Option(help="Number of samples per batch")] = 32,
    num_epochs: Annotated[int, typer.Option(help="Number of epochs for training")] = 20,
    num_samples: Annotated[int, typer.Option(help="Number of trials for tuning")] = 50,
    gpu_per_trial: Annotated[
        float, typer.Option(help="Number of GPU per trial")
    ] = 0.25,
):
    config = {
        "embedding_dim": tune.choice([2**i for i in range(4, 8)]),  # 16, 32, 64, 128
        "num_layers": tune.choice([i for i in range(1, 9)]),  # 1 to 8
        "num_heads": tune.choice([1, 2, 4, 8]),  # 1, 2, 4, 8
        "dropout": tune.quniform(0, 0.2, 0.02),
        "lr": tune.loguniform(1e-5, 1e-1),
        "num_classes": num_classes,
        "val_size": val_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }

    # early stopping with adaptive successive halving
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
    )

    trainable = functools.partial(train_loop, dataset_loc=dataset_loc)
    # make sure GPU is available
    device = get_device()
    if device.type == "cuda":
        trainable = tune.with_resources(trainable, {"gpu": gpu_per_trial})
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(num_samples=num_samples, scheduler=scheduler),
        param_space=config,
    )
    results = tuner.fit()

    save_path = Path(f"runs/{run_id}")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # save tune results
    results_df = results.get_dataframe(filter_metric="val_loss", filter_mode="min")
    results_df.to_csv(save_path / "tune_results.csv", index=False)

    # save best model and params
    best_result = results.get_best_result("val_loss", mode="min")
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        checkpoint_dict = torch.load(Path(checkpoint_dir) / "checkpoint.pt")

        # save model_state to save_path
        model_state = checkpoint_dict["model_state"]
        torch.save(model_state, save_path / f"best_model.pt")

        # save model parameters
        with open(save_path / "args.json", "w") as f:
            json.dump(checkpoint_dict["model_kwargs"], f, indent=4, sort_keys=False)

    return results


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
