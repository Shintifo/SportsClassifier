from pathlib import Path
import os

import torch
from torch.nn.functional import one_hot

class Encoder:
	def __init__(self, path:Path):
		self.path = path / "labels.txt"
		self.values = self.create_encoding(self.path)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def read_labels(self, path):
		if not os.path.exists(path):
			collect_labels()

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
			value = value.to(self.device).int()
			if (enc == value).all():
				return key
		return "No Label"

	def encode(self, item) -> torch.Tensor:
		if item not in self.values.keys():
			raise Exception
		return self.values[item]


def collect_labels():
	"""
	Creates labels.txt required for Encoder, that lists all possible labels.
	"""
	labels = []
	main_dir = Path('datasets')
	for file in os.listdir(main_dir / "train"):
		path = os.path.join(os.path.join(main_dir / "train", file))
		if os.path.isdir(path):
			labels.append(file)

	with open(main_dir / "labels.txt", "w") as f:
		f.write("\n".join(labels))
