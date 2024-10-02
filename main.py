import io

import torch
from fastapi import FastAPI, UploadFile, File
from starlette.responses import RedirectResponse

from model import EffNet
from test import predict

app = FastAPI()

@app.get("/")
async def root():
	return RedirectResponse(url="/docs", status_code=302)

@app.post("/uploadimage")
async def uploadimage(file: UploadFile = File(...)):
	file_content = file.file.read()
	label = classify_sport(io.BytesIO(file_content))
	return label


def classify_sport(file_content):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = EffNet()
	model.to(device)

	checkpoint_path = "base.pth"
	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint['model'])

	label = predict(file_content, model, device, datasets_path="sports")

	return label
