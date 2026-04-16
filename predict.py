import io
import base64
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from cog import BasePredictor, Input


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
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
        self.model = NeuralNetwork().to(self.device)
        self.model.load_state_dict(
            torch.load("model.pth", map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def predict(self, image: str = Input(description="Base64 encoded image")) -> str:
        # Remove data URI prefix if present
        if "," in image:
            image = image.split(",", 1)[1]

        image_data = base64.b64decode(image)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img_tensor)
            predicted_class = self.classes[pred[0].argmax(0)]

        return predicted_class
