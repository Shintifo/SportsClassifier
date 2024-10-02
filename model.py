import torch
import torch.nn as nn
from torchvision import models







class EffNet(nn.Module):
	def __init__(self):
		super(EffNet, self).__init__()
		self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
		# self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

		self.model.classifier =  nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(1280, 100),
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
