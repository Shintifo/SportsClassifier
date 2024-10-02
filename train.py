import argparse
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from dataset import SportsDataset
from model import Model, EffNet
from test import test


def train_epoch(
		model: nn.Module,
		loader: DataLoader,
		optimizer: optim.Optimizer,
		device: torch.device,
		loss_fn,
		epoch: int,
):
	model.train()
	train_loss = 0.0

	epoch_loop = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}", leave=False)

	for x, y in epoch_loop:
		optimizer.zero_grad()
		x, y = x.to(device), y.to(device)

		output = model(x)

		loss = loss_fn(output, y.float())
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		epoch_loop.set_postfix(loss=loss.item())
	print("Epoch {}:".format(epoch))
	print(" Train Loss:", train_loss / len(loader))


def train(
		model: nn.Module,
		optimizer: optim.Optimizer,
		loss: nn.Module,
		train_loader: DataLoader,
		val_loader: DataLoader,
		epochs: int,

):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	best = 0.0
	for epoch in range(1, epochs + 1):
		train_epoch(model, train_loader, optimizer, device, loss, epoch)
		accuracy = test(model=model, loader=val_loader, device=device)

		if accuracy > best:
			save_checkpoint(epoch, model, optimizer)


def save_checkpoint(
		epoch: int,
		model: nn.Module,
		optimizer: optim.Optimizer,
):
	obj = {
		"epoch": epoch,
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
	}
	torch.save(obj, str(f"best.pth"))


def main(path: Path, lr: float, epochs: int, batch_size: int, img_size: int):
	model = EffNet()

	train_set = SportsDataset(img_size=img_size, path=path, set_type='train')
	valid_set = SportsDataset(img_size=img_size, path=path, set_type='valid')

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

	optimizer = optim.Adam(model.parameters(), lr=lr)
	# optimizer = optim.SGD(model.parameters(), lr=lr)
	loss_fn = nn.CrossEntropyLoss()
	train(
		model=model,
		optimizer=optimizer,
		loss=loss_fn,
		train_loader=train_loader,
		val_loader=valid_loader,
		epochs=epochs
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
	parser.add_argument("--batch_size", type=int, default=40, help="Batch size")
	parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
	parser.add_argument("--img_size", type=int, default=128, help="Image size")
	parser.add_argument("--dataset", type=Path, help="Path to data")
	args = parser.parse_args()

	main(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, path=args.dataset, img_size=args.img_size)
