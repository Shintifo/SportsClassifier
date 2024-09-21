import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SportsDataset
from model import Model, EffNet


def HammingDistance(y_pred, y):
	y_pred = (y_pred >= 0.5).float()
	return 1.0 - torch.mean((y.float() != y_pred).float()).item()


def test(
		model: nn.Module,
		loader: DataLoader,
		device: torch.device,
		is_test: bool = False,
):
	dataset = "Test" if is_test else "Valid"
	with torch.no_grad():
		model.eval()
		test_loop = tqdm(loader, total=len(loader), leave=False)
		total = 0
		correct = 0
		for x, y in test_loop:
			x, y = x.to(device), y.to(device)

			output = model(x)

			softmax = nn.Softmax(dim=1)
			output = softmax(output)


			y_pred = (output >= 0.5).float()

			correct += torch.sum(torch.all(y_pred == y, dim=1)).item()
			total += len(y_pred)
			test_loop.set_postfix(accuracy=correct / total)

		print(f" {dataset} accuracy:", correct / total)
		return correct / total


def single_test(
		model: nn.Module,
		test_set,
		index: int,
		device: torch.device,
):
	with torch.no_grad():
		model.eval()

		img, label = test_set[index]

		image = img.permute(1, 2, 0).to(device)
		torch.save(image, "new.jpg")

		img, label = img.to(device), label.to(device)

		img.unsqueeze_(0)
		output = model(img)

	y_pred = (output >= 0.5).int()
	label = test_set.get_label(label)
	predicted = test_set.get_label(y_pred)

	print("Label:",label)
	print("Predicted:", predicted)


def start_test(
		model_type: str,
		checkpoint_path: Path,
		batch_size: int,
		dataset_path: Path,
		img_size: int,
		test_type: str
):

	match model_type:
		case "effnet":
			model = EffNet()
		case _:
			model = Model()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model.to(device)
	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint['model'])

	test_set = SportsDataset(img_size=img_size, path=dataset_path)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

	match test_type:
		case "full":
			test(model, test_loader, device, is_test=True)
		case "single":
			index = random.randint(0, len(test_set) - 1)
			single_test(model=model, test_set=test_set, index=index, device=device)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--test_type", type=str, default="full", help="Could be single or full")
	parser.add_argument("--checkpoint", type=Path, help="path to checkpoint")
	parser.add_argument("--model", type=str, help="model type")
	parser.add_argument("--img_size", type=int, default=156, help="input image size")
	parser.add_argument("--dataset", type=Path, help="path to test set")
	parser.add_argument("--batch_size", type=int, help="Batch size")

	args = parser.parse_args()

	start_test(model_type=args.model, checkpoint_path=args.checkpoint, batch_size=args.batch_size,
			   dataset_path=args.dataset, img_size=args.img_size, test_type=args.test_type)