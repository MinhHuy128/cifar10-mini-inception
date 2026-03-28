import torch
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
from model import MiniInceptionNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

model = MiniInceptionNet(num_classes=10).to(device)
try:
    model.load_state_dict(torch.load('weights/best_model.pth', map_location=device, weights_only=True))
    model.eval()
except FileNotFoundError:
    print("Errors: Not found weights/best_model.pth.")

def predict(img):
    if img is None:
        return None
    
    # PIL -> Tensor and add dimension batch (1, C, H, W)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        # Convert logit output to probability (total = 100%)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    result = {classes[i]: float(probabilities[i]) for i in range(10)}
    return result

#  UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=gr.Label(num_top_classes=3, label="Predictive model:"),
    title="CIFAR-10 Image Recognition with Custom Mini-InceptionNet",
    description="Upload any image and see which class the model predicts it belongs to",
    theme="default"
)

if __name__ == '__main__':
    demo.launch(share=True)