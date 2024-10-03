import random
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils import data
from PIL import Image
from torchvision.transforms import transforms, InterpolationMode

from encoder import Encoder


class SportsDataset(data.Dataset):
	def __init__(self, img_size: int, path: Path, set_type: str = "test"):
		super(SportsDataset, self).__init__()
		self.data = []
		self.encoder = Encoder(path)

		self.load_data(path, f"{set_type}.txt")

		self.transforms = self.collect_transforms(set_type, img_size)
		self.normalize = transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)

	def __getitem__(self, index: int):
		image_path = self.data[index]["img"]

		img = Image.open(image_path).convert('RGB')
		img = self.transforms(img)
		img = self.normalize(img)

		#Its folder is a label
		label = image_path.split(os.sep)[-2]

		idx = self.encoder.encode(label)
		enc_label = torch.zeros(self.encoder.num_classes)
		enc_label[idx] = 1

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

	def get_label(self, enc_label) -> str:
		return self.encoder.decode(enc_label)

	@staticmethod
	def collect_transforms(set_type, img_size=128):
		if set_type == "train":
			return transforms.Compose([
				transforms.RandomVerticalFlip(p=0.3),
				transforms.RandomHorizontalFlip(),
				transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
				transforms.ToTensor()
			])

		return transforms.Compose([
			transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
			transforms.ToTensor()
		])

	@staticmethod
	def __collect__(path: Path):
		"""
		Creates files with paths to images, split by groups
		:param path: Path to whole dataset
		"""
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
if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--path', type=Path, help='path to dataset')
	args = parser.parse_args()

	SportsDataset.__collect__(args.path)

