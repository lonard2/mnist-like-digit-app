from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torchvision.transforms as transforms
import base64
from io import BytesIO
from PIL import Image
import os

# Load model
from train_generator import Generator

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = Generator()
model.load_state_dict(torch.load("models/generator.pth", map_location="cpu"))
model.eval()

def generate_images(label: int, num_samples=5):
    z = torch.randn(num_samples, 100)
    labels = torch.tensor([label] * num_samples)
    with torch.no_grad():
        images = model(z, labels).detach().cpu()
    images = (images + 1) / 2  # normalize from [-1,1] to [0,1]
    return images

def to_base64(img_tensor):
    img_pil = transforms.ToPILImage()(img_tensor.squeeze(0))
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, digit: int = Form(...)):
    imgs = generate_images(digit)
    encoded_images = [to_base64(img.unsqueeze(0)) for img in imgs]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "images": encoded_images,
        "selected_digit": digit
    })