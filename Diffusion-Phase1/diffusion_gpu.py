import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from dirty_documents_loader import create_dirty_documents_dataloader

# dirty_root = r"D:\path\to\dirty_documents_dataset"

# train_loader = create_dirty_documents_dataloader(
#     root_dir=dirty_root,
#     batch_size=2,
#     target_size=(256,256),
#     split='train',
#     add_noise=True,
#     noise_type='mixed'
# )


# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths (Mac Compatible)
base_dir = os.path.dirname(os.path.abspath(__file__))
noisy_images_dir = os.path.join(base_dir, 'dataset', 'noisy_images')
denoised_images_dir = os.path.join(base_dir, 'dataset', 'Denoised_images')
output_dir = os.path.join(base_dir, 'dataset', 'diffusion_output')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("GPU-accelerated image denoising using diffusion model")

# Neural network model for denoising with fixed dimensions
class DenoisingUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DenoisingUNet, self).__init__()
        
        # Encoder (downsampling)
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
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 = 128 + 128 (skip connection)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 + 64 (skip connection)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Save input dimensions
        input_size = x.size()
        
        # Encoder
        e1 = self.enc1_conv(x)
        e1_pool = self.pool1(e1)
        
        e2 = self.enc2_conv(e1_pool)
        e2_pool = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(e2_pool)
        
        # Decoder with skip connections
        d1 = self.upconv1(b)
        
        # Ensure dimensions match for skip connection
        # If sizes don't match, resize d1 to match e2
        if d1.size() != e2.size():
            d1 = F.interpolate(d1, size=(e2.size(2), e2.size(3)), mode='bilinear', align_corners=True)
            
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1_conv(d1)
        
        d2 = self.upconv2(d1)
        
        # Ensure dimensions match for skip connection
        # If sizes don't match, resize d2 to match e1
        if d2.size() != e1.size():
            d2 = F.interpolate(d2, size=(e1.size(2), e1.size(3)), mode='bilinear', align_corners=True)
            
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2_conv(d2)
        
        # Final output
        output = self.output_conv(d2)
        
        # Ensure output has the same dimensions as input
        if output.size() != input_size:
            output = F.interpolate(output, size=(input_size[2], input_size[3]), mode='bilinear', align_corners=True)
        
        # Apply sigmoid to constrain values between 0 and 1
        output = torch.sigmoid(output)
        
        return output

# Dataset for loading image pairs
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, target_size=(256, 256)):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.target_size = target_size  # Target size for resizing
        self.image_files = [f for f in os.listdir(noisy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load noisy image
        noisy_path = os.path.join(self.noisy_dir, img_name)
        noisy_img = cv2.imread(noisy_path)
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB) / 255.0
        
        # Load clean image if available
        clean_path = os.path.join(self.clean_dir, img_name)
        if os.path.exists(clean_path):
            clean_img = cv2.imread(clean_path)
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB) / 255.0
        else:
            clean_img = noisy_img.copy()
        
        # Resize both images to the same dimensions
        noisy_img = cv2.resize(noisy_img, self.target_size, interpolation=cv2.INTER_AREA)
        clean_img = cv2.resize(clean_img, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to tensors
        noisy_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1)).float()
        clean_tensor = torch.from_numpy(clean_img.transpose(2, 0, 1)).float()
        
        return noisy_tensor, clean_tensor, img_name

# GPU-accelerated diffusion denoising class
class DiffusionDenoiser:
    def __init__(self, timesteps=30):
        self.timesteps = timesteps
        self.device = device
        
        # Initialize neural network model
        self.model = DenoisingUNet().to(self.device)
        
    def train(self, noisy_dir, clean_dir, batch_size=1, num_epochs=5, learning_rate=0.001, target_size=(256, 256)):
        """Train the neural network model on image pairs"""
        # Set model to training mode
        self.model.train()
        
        # Create dataset and dataloader with consistent image sizes
        dataset = DenoisingDataset(noisy_dir, clean_dir, target_size=target_size)
        
        # Use a smaller batch size if images are large
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for noisy_imgs, clean_imgs, _ in progress_bar:
                # Move data to device
                noisy_imgs = noisy_imgs.to(self.device)
                clean_imgs = clean_imgs.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save the trained model
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'denoising_model.pth'))
        print(f"Model saved to {os.path.join(output_dir, 'denoising_model.pth')}")
        
        # Set model back to evaluation mode
        self.model.eval()
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {model_path}")
    
    def denoise_image(self, noisy_image, progress_callback=None, target_size=(256, 256)):
        """Apply GPU-accelerated denoising"""
        # Make sure image is in [0, 1] range
        if noisy_image.max() > 1.0:
            noisy_image = noisy_image / 255.0
            
        # Store original dimensions for later resizing back
        original_h, original_w = noisy_image.shape[:2]
        
        # Resize for model processing
        noisy_image_resized = cv2.resize(noisy_image, target_size, interpolation=cv2.INTER_AREA)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Convert to tensor and move to device
        noisy_tensor = torch.from_numpy(noisy_image_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        
        # Process through neural network
        with torch.no_grad():
            denoised_tensor = self.model(noisy_tensor)
        
        # Convert back to numpy
        denoised_image = denoised_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        
        # Apply post-processing enhancement
        denoised_image = self.enhance_details(denoised_image)
        
        # Resize back to original dimensions
        denoised_image = cv2.resize(denoised_image, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        
        return denoised_image
    
    def enhance_details(self, image):
        """Enhance image details after denoising"""
        # Ensure image is float32 (not float64) for OpenCV compatibility
        if image.dtype == np.float64:
            image = image.astype(np.float32)
            
        # Convert to uint8 for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        clahe_img = cv2.merge((cl, a, b))
        clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        # Apply subtle sharpening
        kernel = np.array([[-1, -1, -1], 
                          [-1, 9, -1], 
                          [-1, -1, -1]]) / 9.0
        sharpened = cv2.filter2D(img_uint8, -1, kernel).astype(np.float32) / 255.0
        
        # Blend for final result
        enhanced = 0.7 * image + 0.2 * clahe_img + 0.1 * sharpened
        return np.clip(enhanced, 0, 1).astype(np.float32)

# Function to load and preprocess an image
def load_image(image_path):
    """Load and preprocess an image"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to RGB and normalize to [0, 1] as float32 (not float64)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image

# Function to save an image
def save_image(image, output_path):
    """Save an image to disk"""
    # Ensure image is in the right format for OpenCV
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    
    # Convert to uint8 for OpenCV
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
    print(f"Saved image to {output_path}")

# Function to visualize denoising progress
def visualize_progress(i, total_steps, current_image):
    """Display progress during denoising"""
    if i % 5 == 0 or i == total_steps - 1:
        print(f"Denoising step {i+1}/{total_steps}")

# Function to calculate PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    # Ensure images are in the same range [0, 1] or [0, 255]
    if img1.max() <= 1.0 and img2.max() > 1.0:
        img2 = img2 / 255.0
    elif img2.max() <= 1.0 and img1.max() > 1.0:
        img1 = img1 / 255.0
    
    # Ensure images are float32 for consistent calculations
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Function to test denoising on a single image
def test_on_image(denoiser, image_path, output_path):
    """Test denoising on a single image"""
    # Load the image
    try:
        img = load_image(image_path)
        print(f"Loaded image from {image_path}, shape: {img.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Start denoising
    print("Starting GPU-accelerated denoising process...")
    start_time = time.time()
    
    # Apply denoising
    denoised_img = denoiser.denoise_image(img)
    
    end_time = time.time()
    print(f"Denoising completed in {end_time - start_time:.2f} seconds")
    
    # Save the result
    save_image(denoised_img, output_path)
    
    # Create comparison visualization
    plt.figure(figsize=(12, 6))
    
    # Ensure images are in the right format for matplotlib
    noisy_display = np.clip(img, 0, 1)
    denoised_display = np.clip(denoised_img, 0, 1)
    
    plt.subplot(1, 2, 1)
    plt.imshow(noisy_display)
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_display)
    plt.title('Denoised Image')
    plt.axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(os.path.dirname(output_path), 'comparison.png')
    plt.savefig(comparison_path)
    print(f"Saved comparison to {comparison_path}")
    
    return denoised_img

# Function to train on all images in the folder
def train_on_all_images(denoiser, noisy_dir, clean_dir, output_dir, target_size=(256, 256)):
    """Train the denoiser on all images in the folder using GPU acceleration"""
    print("\nTraining on all images in the folder using GPU acceleration...")
    
    # Get all image files
    image_files = [f for f in os.listdir(noisy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images for training")
    
    # Create a results directory
    results_dir = os.path.join(output_dir, 'gpu_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Train the model with consistent image sizes
    denoiser.train(noisy_dir, clean_dir, batch_size=1, num_epochs=5, learning_rate=0.001, target_size=target_size)
    
    # Process each image and evaluate
    print("\nEvaluating trained model on all images...")
    all_psnr = []
    processing_times = []
    
    for img_file in tqdm(image_files):
        try:
            # Load noisy image
            noisy_path = os.path.join(noisy_dir, img_file)
            noisy_img = load_image(noisy_path)
            
            # Check if clean reference exists
            clean_path = os.path.join(clean_dir, img_file)
            has_clean_reference = os.path.exists(clean_path)
            
            if has_clean_reference:
                clean_img = load_image(clean_path)
            
            # Process the image
            start_time = time.time()
            denoised_img = denoiser.denoise_image(noisy_img, target_size=target_size)
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Save the denoised image
            output_path = os.path.join(results_dir, f'denoised_{img_file}')
            save_image(denoised_img, output_path)
            
            # Calculate metrics if clean reference exists
            if has_clean_reference:
                psnr = calculate_psnr(denoised_img, clean_img)
                all_psnr.append(psnr)
                
                # Create comparison visualization
                plt.figure(figsize=(15, 5))
                
                # Ensure images are in the right format for matplotlib
                noisy_display = np.clip(noisy_img, 0, 1)
                denoised_display = np.clip(denoised_img, 0, 1)
                clean_display = np.clip(clean_img, 0, 1)
                
                plt.subplot(1, 3, 1)
                plt.imshow(noisy_display)
                plt.title('Noisy Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(denoised_display)
                plt.title(f'Denoised (PSNR: {psnr:.2f}dB)')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(clean_display)
                plt.title('Ground Truth (Clean)')
                plt.axis('off')
                
                plt.tight_layout()
                comparison_path = os.path.join(results_dir, f'comparison_{img_file}')
                plt.savefig(comparison_path)
                plt.close()
                
                print(f"Processed {img_file}: PSNR = {psnr:.2f}dB, Time = {processing_time:.2f}s")
            else:
                print(f"Processed {img_file}: Time = {processing_time:.2f}s (no clean reference)")
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Calculate and report overall metrics
    if all_psnr:
        avg_psnr = np.mean(all_psnr)
        avg_time = np.mean(processing_times)
        print(f"\nEvaluation completed on {len(image_files)} images")
        print(f"Average PSNR: {avg_psnr:.2f}dB")
        print(f"Average processing time: {avg_time:.2f}s per image")
        
        # Save metrics to file
        with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Total images processed: {len(image_files)}\n")
            f.write(f"Images with clean reference: {len(all_psnr)}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f}dB\n")
            f.write(f"Average processing time: {avg_time:.2f}s per image\n")
            f.write("\nPer-image PSNR values:\n")
            for i, (img_file, psnr) in enumerate(zip(image_files[:len(all_psnr)], all_psnr)):
                f.write(f"{img_file}: {psnr:.2f}dB\n")
    
    return results_dir

# Main function
def main():
    print("Starting GPU-accelerated diffusion denoising")
    
    # Create denoiser
    denoiser = DiffusionDenoiser(timesteps=30)
    
    # Train on all images in the folder with consistent image sizes
    target_size = (256, 256)  # Use a smaller size that works well with the UNet architecture and fits in memory
    results_dir = train_on_all_images(denoiser, noisy_images_dir, denoised_images_dir, output_dir, target_size)
    print(f"Results saved to {results_dir}")
    
    # Also test on a specific image
    test_image = os.path.join(noisy_images_dir, '00000004.png')
    output_image = os.path.join(output_dir, 'gpu_denoised_00000004.png')
    
    if os.path.exists(test_image):
        print("\nPerforming detailed analysis on test image...")
        denoised = test_on_image(denoiser, test_image, output_image)
        print(f"Denoised image saved to {output_image}")

# Run the main function
if __name__ == "__main__":
    main()