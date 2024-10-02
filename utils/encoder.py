import numpy as np
import torch
from torch.nn.functional import one_hot
import os

class Encoder:
	def __init__(self, path='sports/labels.txt'):
		self.values = self.create_encoding(path)

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
	for file in os.listdir('sports/train'):
		path = os.path.join(os.path.join('sports/train', file))
		if os.path.isdir(path):
			labels.append(file)

	with open("sports/labels.txt", "w") as f:
		f.write("\n".join(labels))
	print()

if __name__ == '__main__':
	e = Encoder().values
	print(e)
