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


class TrafficSignsDataset(data.Dataset):
	def __init__(self, img_size: int, is_train: bool, path: Path):
		super(TrafficSignsDataset, self).__init__()
		self.dir = path
		self.data = self.load_data(path, ("train" if is_train else "test"))
		self.is_train = is_train

		if is_train:
			self.transforms = transforms.Compose([
				# v2.RandomVerticalFlip(p=0.3),
				v2.RandomHorizontalFlip(),
				# v2.ColorJitter(contrast=0.3, brightness=0.3, hue=0.1),
				v2.RandomAdjustSharpness(sharpness_factor=3),
				v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
				# v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
			])
		else:
			self.transforms = transforms.Compose([
				v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
				# v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
			])

		self.normalize = transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)

	def __getitem__(self, index: int):
		image = self.data[index]["img"]
		img_path = self.dir / "images" / image

		img = Image.open(img_path).convert('RGB')
		img = self.transforms(img)
		img = self.normalize(img)

		return img, self.data[index]['label']

	def __len__(self):
		return len(self.data)

	def load_data(self, path: Path, item_list: str):
		data = []
		with open(path / item_list, "r") as f:
			lines = f.readlines()

		for i, row in enumerate(lines):
			img, label = row.split(":")
			label = list(map(int, label.split()))
			data.append({
				"img": img,
				"label": torch.tensor(label)
			})
		return data

	@staticmethod
	def __save__(path: Path, is_train: bool, images: list[str]):
		filepath = path / ("train" if is_train else "test")
		with open(filepath, "w") as f:
			for img in images:
				f.write(img)

	@staticmethod
	def create_dataset(path: Path, ratio: float = 0.8):
		random.seed(337)
		with open(path / "data", "r") as f:
			items = f.readlines()

		index = int(len(items) * ratio)
		train_set = items[:index]
		test_set = items[index:]

		random.shuffle(train_set)
		random.shuffle(test_set)

		TrafficSignsDataset.__save__(path, True, train_set)
		TrafficSignsDataset.__save__(path, False, test_set)

	@staticmethod
	def generate_labels(input):
		labels = {
			"trafficlight": 0,
			"speedlimit": 1,
			"stop": 2,
			"crosswalk": 3
		}
		data = torch.tensor([labels[i] for i in input])

		enc = one_hot(data, num_classes=len(labels))
		enc = enc.sum(dim=0)
		enc = torch.clamp(enc, max=1)
		label = " ".join(map(str, enc.tolist()))
		return label

	@staticmethod
	def convert_from_xml(path: Path):
		img_path = path / "images"
		images = [img for img in os.listdir(img_path)]

		labels_path = path / "annotations"
		labels = []
		for file in os.listdir(labels_path):
			root = ET.parse(labels_path / file).getroot()
			names = [name.text for name in root.findall('object/name')]
			item_label = TrafficSignsDataset.generate_labels(names)
			labels.append(item_label)

		items = [f"{images[i]}:{labels[i]}\n" for i in range(len(images))]
		random.seed(337)
		random.shuffle(items)
		with open(path / "data", "w") as f:
			for item in items:
				f.write(item)


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
