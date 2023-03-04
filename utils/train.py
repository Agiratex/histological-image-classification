import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from utils.compute_metrics import compute_metrics
import gc


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One training cycle (loop).
    """

    model.train()

    epoch_loss = []
    batch_metrics_list = defaultdict(list)

    for i, (input, labels) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="loop over train batches",
    ):

        input, labels = input.to(device), labels.to(device)
        # YOUR CODE HERE
        # Подсчет лосса и шаг оптимизатора
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar(
            "batch loss / train", loss.item(), epoch * len(dataloader) + i
        )

        with torch.no_grad():
            model.eval()
            outputs_inference = model(input)
            model.train()

        batch_metrics = compute_metrics(
            outputs=outputs_inference,
            labels=labels,
        )

        for metric_name, metric_value in batch_metrics.items():
            batch_metrics_list[metric_name].append(metric_value)
            writer.add_scalar(
                f"batch {metric_name} / train",
                metric_value,
                epoch * len(dataloader) + i,
            )

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    writer.add_scalar("loss / train", avg_loss, epoch)

    for metric_name, metric_value_list in batch_metrics_list.items():
        metric_value = np.mean(metric_value_list)
        print(f"Train {metric_name}: {metric_value}\n")
        writer.add_scalar(f"{metric_name} / train", metric_value, epoch)

def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One evaluation cycle (loop).
    """

    model.eval()

    epoch_loss = []
    batch_metrics_list = defaultdict(list)

    with torch.no_grad():

        for i, (input, labels) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
        ):

            input, labels = input.to(device), labels.to(device)
            
            outputs = model(input)
            loss = criterion(outputs, labels)

            epoch_loss.append(loss.item())
            writer.add_scalar(
                "batch loss / test", loss.item(), epoch * len(dataloader) + i
            )

            batch_metrics = compute_metrics(
                outputs=outputs,
                labels=labels,
            )

            for metric_name, metric_value in batch_metrics.items():
                batch_metrics_list[metric_name].append(metric_value)
                writer.add_scalar(
                    f"batch {metric_name} / test",
                    metric_value,
                    epoch * len(dataloader) + i,
                )

        avg_loss = np.mean(epoch_loss)
        print(f"Test loss:  {avg_loss}\n")
        writer.add_scalar("loss / test", avg_loss, epoch)

        for metric_name, metric_value_list in batch_metrics_list.items():
            metric_value = np.mean(metric_value_list)
            print(f"Test {metric_name}: {metric_value}\n")
            writer.add_scalar(f"{metric_name} / test", np.mean(metric_value), epoch)

def train(
    n_epochs: int,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    path : str
) -> None:
    """
    Training loop.
    """

    for epoch in range(n_epochs):

        print(f"Epoch [{epoch+1} / {n_epochs}]\n")

        train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            writer=writer,
            device=device,
            epoch=epoch,
        )
        evaluate_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            writer=writer,
            device=device,
            epoch=epoch,
        )
        torch.save(model.state_dict(), f"{path}{epoch}.pth")
        gc.collect()

if __name__ == '__main__':
    pass
