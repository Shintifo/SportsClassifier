import io

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from dataset import SportsDataset
from model import EffNet
from utils.encoder import Encoder
from test import predict

app = FastAPI()


@app.post("/uploadimage")
async def uploadimage(file: UploadFile = File(...)):
	file_content = file.file.read()

	label = classify_sport(io.BytesIO(file_content))

	return label

	# return JSONResponse(content={
	#     "filename": file_content,
	#     "file_content_type": file.content_type,
	#     "filesize": len(file_content),
	#     "label": label
	# })


def classify_sport(file_content):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = EffNet()
	model.to(device)

	checkpoint_path = "base.pth"
	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint['model'])

	label = predict(file_content, model, device)

	return label
