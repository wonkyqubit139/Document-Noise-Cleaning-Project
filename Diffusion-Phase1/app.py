import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os

# --- 1. CONFIGURATION ---
device = torch.device('cpu')
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'dataset', 'diffusion_output', 'denoising_model.pth')

# --- 2. DEFINE THE MODEL (Same as before) ---
class DenoisingUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DenoisingUNet, self).__init__()
        # Encoder
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.size()
        e1 = self.enc1_conv(x)
        e1_pool = self.pool1(e1)
        e2 = self.enc2_conv(e1_pool)
        e2_pool = self.pool2(e2)
        b = self.bottleneck(e2_pool)
        d1 = self.upconv1(b)
        if d1.size() != e2.size():
            d1 = F.interpolate(d1, size=(e2.size(2), e2.size(3)), mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1_conv(d1)
        d2 = self.upconv2(d1)
        if d2.size() != e1.size():
            d2 = F.interpolate(d2, size=(e1.size(2), e1.size(3)), mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2_conv(d2)
        output = self.output_conv(d2)
        if output.size() != input_size:
            output = F.interpolate(output, size=(input_size[2], input_size[3]), mode='bilinear', align_corners=True)
        output = torch.sigmoid(output)
        return output

# --- 3. LOAD THE BRAIN ---
print("Loading model for the App...")
model = DenoisingUNet().to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    print("WARNING: Model not found. App will output garbage.")

# --- 4. THE CLEANING FUNCTION ---
def clean_document(image):
    if image is None:
        return None
    
    # 1. Resize input to 256x256
    original_h, original_w = image.shape[:2]
    img_resized = cv2.resize(image, (256, 256))
    
    # 2. Normalize and Convert to Tensor
    img_float = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    # 3. Predict
    with torch.no_grad():
        output = model(tensor)
    
    # 4. Post-process
    output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    
    # 5. Resize back to original size for better viewing
    final_output = cv2.resize(output_np, (original_w, original_h))
    
    return final_output

# --- 5. LAUNCH THE WEB UI ---
interface = gr.Interface(
    fn=clean_document,
    inputs=gr.Image(label="Drop Dirty Document Here"),
    outputs=gr.Image(label="Cleaned Document"),
    title="Document Cleaner AI ✨",
    description="Upload a noisy/stained document, and the AI will clean it for you."
)

if __name__ == "__main__":
    interface.launch()