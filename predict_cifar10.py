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
            nn.Linear(3 * 32 * 32, 512),
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
        self.classes = ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"]
        self.model = NeuralNetwork().to(self.device)
        self.model.load_state_dict(
            torch.load("model_cifar10.pth", map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def predict(self, image: str = Input(description="Base64 encoded image")) -> str:
        if "," in image:
            image = image.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img_tensor)
            return self.classes[pred[0].argmax(0)]
