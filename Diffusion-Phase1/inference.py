import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from tqdm import tqdm

# --- 1. SETUP PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(base_dir, 'dataset', 'test')
output_dir = os.path.join(base_dir, 'dataset', 'test_results')
model_path = os.path.join(base_dir, 'dataset', 'diffusion_output', 'denoising_model.pth')

os.makedirs(output_dir, exist_ok=True)
device = torch.device('cpu') # Force CPU for Mac

print(f"Looking for model at: {model_path}")
print(f"Looking for test images at: {test_dir}")

# --- 2. DEFINE THE MODEL ARCHITECTURE (Must match training exactly) ---
class DenoisingUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DenoisingUNet, self).__init__()
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
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

# --- 3. RUN INFERENCE ---
def main():
    # Load Model
    if not os.path.exists(model_path):
        print("Error: Model file not found! Did you finish training?")
        return

    print("Loading model...")
    model = DenoisingUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get Images
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in test folder.")

    # Loop
    for img_name in tqdm(image_files):
        img_path = os.path.join(test_dir, img_name)
        
        # Load and Preprocess
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        
        # Resize to 256x256 for the model
        img_resized = cv2.resize(original_img, (256, 256))
        img_normalized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convert to Tensor
        tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        
        # Denoise
        with torch.no_grad():
            output = model(tensor)
            
        # Post-process
        output_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        # Optional: Resize back to original size? 
        # For now, let's keep it 256x256 so it looks clean, or resize back:
        output_img = cv2.resize(output_img, (original_img.shape[1], original_img.shape[0]))
        
        # Save
        save_path = os.path.join(output_dir, f"cleaned_{img_name}")
        cv2.imwrite(save_path, output_img)

    print(f"\nSuccess! Check your results in: {output_dir}")

if __name__ == "__main__":
    main()