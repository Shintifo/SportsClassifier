import torch
import torch.nn as nn
from torchvision import models


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.conv_s = nn.Sequential(
			nn.Sequential(
				nn.Conv2d(3, 32, kernel_size=3, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2, stride=3)
			),
			nn.Sequential(
				nn.Conv2d(32, 64, kernel_size=2, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2, stride=3)
			),
			nn.Sequential(
				nn.Conv2d(64, 128, kernel_size=2, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2, stride=3)
			),
		)

		self.flatten = nn.Flatten()

		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(4608, 1024),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(1024, 100),
		)

	def forward(self, x):
		x = self.conv_s(x)
		x = self.flatten(x)
		x = self.classifier(x)
		return x







class EffNet(nn.Module):
	def __init__(self):
		super(EffNet, self).__init__()
		# self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
		self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

		self.model.classifier =  nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(1792, 100),

		)

	def forward(self, x):
		x = self.model(x)
		return x

def test(model, size):
	print("Here!")
	dummy_input = torch.randn(1, 3, size, size)
	output = model(dummy_input)
	print(output)


if __name__ == '__main__':
	print(EffNet().model)
	# test(EffNet(), 128)
