import argparse

import torch
from model import EffNet


def wrap_model(checkpoint):
	model = EffNet(load_effnet_weights=False, export=True)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	checkpoint = torch.load(checkpoint, weights_only=False, map_location=device)
	model.load_state_dict(checkpoint['model'])
	model.to(device)
	model.eval()

	dummy_input = torch.randn(1, 3, 128, 128).to(device)
	torch.onnx.dynamo_export(model, dummy_input).save("model.onnx")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', type=str, default='base.pth')
	args = parser.parse_args()
	wrap_model(args.checkpoint)
