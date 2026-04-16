import io
import base64
import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps
from cog import BasePredictor, Input


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cpu"
        self.classes = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]
        self.model = NeuralNetwork(hidden_size=512).to(self.device)
        self.model.load_state_dict(
            torch.load("model_optimized.pth", map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def predict(self, image: str = Input(description="Base64 encoded image")) -> str:
        if "," in image:
            image = image.split(",", 1)[1]

        image_data = base64.b64decode(image)
        img = Image.open(io.BytesIO(image_data)).convert("L")
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img_tensor)
            predicted_class = self.classes[pred[0].argmax(0)]

        return predicted_class
