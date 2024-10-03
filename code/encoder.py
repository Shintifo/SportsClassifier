from pathlib import Path
import os
import numpy as np



class Encoder:
	def __init__(self, path: Path):
		self.path = path / "labels.txt"
		self.num_classes = None
		self.indices = None
		self.values = None
		self.create_encoding(self.path)

	def read_labels(self, path) -> list[str]:
		if not os.path.exists(path):
			collect_labels()

		with open(path, 'r') as f:
			lines = [line.strip() for line in f.readlines()]
		return lines

	def create_encoding(self, path) -> None:
		values = self.read_labels(path)

		self.num_classes = len(values)
		numerical_encode = np.eye(self.num_classes)
		enc_class = np.argmax(numerical_encode, axis=1).tolist()

		self.values = dict(zip(values, enc_class))
		self.indices = dict(zip(enc_class, values))

	def decode(self, idx: int) -> str | None:
		if idx not in self.indices.keys():
			return None
		return self.indices[idx]

	def encode(self, item: str) -> int:
		if item not in self.values.keys():
			raise ValueError
		return self.values[item]


def collect_labels():
	"""
	Creates labels.txt required for Encoder, that lists all possible labels.
	"""
	labels = []
	main_dir = Path('sports')
	for file in os.listdir(main_dir / "train"):
		path = os.path.join(os.path.join(main_dir / "train", file))
		if os.path.isdir(path):
			labels.append(file)

	with open(main_dir / "labels.txt", "w") as f:
		f.write("\n".join(labels))
