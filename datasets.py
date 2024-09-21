import random
import os
import re
from argparse import ArgumentParser
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils import data
from torch.nn.functional import one_hot
from torchvision.transforms import v2, transforms, InterpolationMode

from utils.encoder import Encoder


class SportsDataset(data.Dataset):
	def __init__(self, img_size: int, path: Path, set_type: str = "test"):
		super(SportsDataset, self).__init__()
		self.data = []
		self.encoder = None
		self.path = path

		self.load_data(path, f"{set_type}.txt")

		if set_type == "train":
			self.transforms = transforms.Compose([
				transforms.RandomVerticalFlip(p=0.3),
				transforms.RandomHorizontalFlip(),
				transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
				transforms.ToTensor()
			])
		else:
			self.transforms = transforms.Compose([
				transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
				transforms.ToTensor()
			])

		self.normalize = transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)

	def __getitem__(self, index: int):
		image = self.data[index]["img"]

		img = Image.open(image).convert('RGB')
		img = self.transforms(img)
		img = self.normalize(img)

		label = image.split(os.sep)[-2]
		enc_label = self.encoder.encode(label)

		return img, enc_label

	def __len__(self):
		return len(self.data)

	def load_data(self, path: Path, item_list: str):
		labels = set()
		with open(path / item_list, "r") as f:
			lines = f.readlines()

		for i, img in enumerate(lines):
			label = img.split(os.sep)[-2]
			labels.add(label)
			self.data.append({
				"img": img.strip(),
				"label": label
			})

		self.encoder = Encoder(list(labels))

	def get_label(self, enc_label) -> str:
		return self.encoder.decode(enc_label)

	@staticmethod
	def __collect__(path: Path):
		random.seed(337)
		data = {
			"train": [],
			"test": [],
			"valid": []
		}
		for dirpath, dirnames, filenames in os.walk(path):
			if len(dirnames) != 0:
				continue

			# Train, test, val
			subset = dirpath.split(os.sep)[-2]

			# We reach folder with images
			items = [f"{dirpath}/{file}\n" for file in filenames if file.endswith("jpg")]
			data[subset].extend(items)

		for subset in data.keys():
			random.shuffle(data[subset])
			with open(f"{path}/{subset}.txt", "w") as f:
				[f.write(item) for item in data[subset]]


# TODO updating dataset
# TODO stratify data
# TODO One-Hot Encoding


def test_transform():
	img_size = 156
	t = transforms.Compose([
		v2.RandomVerticalFlip(p=0.3),
		v2.RandomHorizontalFlip(),
		v2.ColorJitter(contrast=0.3, brightness=0.3, hue=0.1),
		v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
		# v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
	])

	img = Image.open("img.jpg").convert('RGB')

	img = t(img)
	img.save("new.jpg")


def count_labels():
	a = set()

	with open("train.csv", "r") as f:
		lines = f.readlines()
	num = len(lines)
	for line in lines:
		name, label = line.strip().split(",")
		a.add(label)
		if label in ['unknown', 'smoke']:
			num -= 1
	print(a)
	print(len(a))
	print(num)


if __name__ == '__main__':
	parser = ArgumentParser()
	subparsers = parser.add_subparsers(dest='mode')

	convert_parser = subparsers.add_parser('collect')
	convert_parser.add_argument('--path', type=Path, help='path to dataset')

	create_parser = subparsers.add_parser('create')
	create_parser.add_argument('--path', type=Path, help='path to dataset')
	create_parser.add_argument('--ratio', type=float, default=0.8)

	convert_parser = subparsers.add_parser('test')

	args = parser.parse_args()

	match args.mode:
		case 'collect':
			SportsDataset.__collect__(args.path)
		case 'test':
			test_transform()
