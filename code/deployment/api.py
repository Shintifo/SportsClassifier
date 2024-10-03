import io
from pathlib import Path

import uvicorn
import numpy as np
import onnxruntime
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from starlette.responses import RedirectResponse

from encoder import Encoder

app = FastAPI()


@app.get("/")
async def root():
	return RedirectResponse(url="/docs", status_code=302)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
	file_content = await file.read()
	label = run_onnx(io.BytesIO(file_content))
	return {"prediction": label}


def image_preprocess(raw_bytes: bytes):
	img = Image.open(raw_bytes)
	img = img.resize((128, 128), Image.BILINEAR)
	img = np.array(img).astype(np.float32)
	img = img / 255.0
	img = img.transpose(2, 0, 1)
	img = np.expand_dims(img, axis=0)
	return img


def run_onnx(file_content):
	img = image_preprocess(file_content)
	ort_session = onnxruntime.InferenceSession("model.onnx")

	input_name = ort_session.get_inputs()[0].name
	prediction = ort_session.run(None, {input_name: img})[0]

	idx = int(np.argmax(prediction[0]))
	encoder = Encoder(Path(""))
	pred_class = encoder.decode(idx)

	return pred_class

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)