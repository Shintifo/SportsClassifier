import numpy as np
import torch
from torch.nn.functional import one_hot

class Encoder:
	def __init__(self, values: list[str]):
		self.values = self.create_encoding(values)


	@staticmethod
	def create_encoding(values: list[str]) -> dict[str, torch.Tensor]:
		num_classes = len(values)
		numerical_encode = torch.tensor(range(num_classes))
		enc_classes = one_hot(numerical_encode, num_classes=num_classes)
		return dict(zip(values, enc_classes))

	def decode(self, enc):
		for key, value in self.values.items():
			value = value.to("cuda" if torch.cuda.is_available() else "cpu").int()
			if (enc == value).all():
				print(np.where(enc.detach().cpu().numpy() == 1), key)
				print(np.where(value.detach().cpu().numpy() == 1), key)
				return key
		return "Error"

	def encode(self, item) -> torch.Tensor:
		return self.values[item]

	def encodings(self) -> dict[str, torch.Tensor]:
		return self.values

if __name__ == '__main__':
	l = ['a', 'b', 'c', 'd', 'e']
	encoder = Encoder(l)

	print(encoder.encodings())
	print(encoder.encode('a'))
