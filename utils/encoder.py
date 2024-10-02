from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import one_hot
import os

class Encoder:
	def __init__(self, path:Path):
		self.path = path / "labels.txt"
		self.values = self.create_encoding(self.path)

	def read_labels(self, path):
		with open(path, 'r') as f:
			lines = [line.strip() for line in f.readlines()]
		return lines

	def create_encoding(self, path) -> dict[str, torch.Tensor]:
		values = self.read_labels(path)

		num_classes = len(values)
		numerical_encode = torch.tensor(range(num_classes))
		enc_classes = one_hot(numerical_encode, num_classes=num_classes)
		return dict(zip(values, enc_classes))

	def decode(self, enc):
		for key, value in self.values.items():
			value = value.to("cuda" if torch.cuda.is_available() else "cpu").int()
			if (enc == value).all():
				return key
		return "Error"

	def encode(self, item) -> torch.Tensor:
		if item not in self.values.keys():
			raise Exception
		return self.values[item]

	def encodings(self) -> dict[str, torch.Tensor]:
		return self.values


def collect_labels():
	labels = []
	main_dir = Path('datasets')
	for file in os.listdir(main_dir / "train"):
		path = os.path.join(os.path.join(main_dir / "train", file))
		if os.path.isdir(path):
			labels.append(file)

	with open(main_dir / "labels.txt", "w") as f:
		f.write("\n".join(labels))
	print()

if __name__ == '__main__':
	e = Encoder().values
	print(e)
