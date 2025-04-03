from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # 🔹 Import CORS Middleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import models_vit
import io
import uvicorn

app = FastAPI()

# 🔹 Enable CORS to allow requests from your frontend on port 5500
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # 🔹 Allow requests from Live Server
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "IDRID2.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

model = models_vit.VisionTransformer(embed_dim=1024, num_heads=16, depth=22, num_classes=5)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

categories = ["anoDR", "bmildDR", "cmoderateDR", "dsevereDR", "eproDR"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        return JSONResponse(content={
            "category": categories[predicted_class.item()],
            "confidence": f"{confidence.item() * 100:.2f}%"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
