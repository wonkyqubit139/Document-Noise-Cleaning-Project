# dirty_documents_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np

# A function to add noise, tailored to the 'mixed' type you requested
def add_mixed_noise(image):
    """Adds a mixture of Gaussian and Salt-and-Pepper noise to an image."""
    img = image.copy()
    
    # 1. Gaussian Noise (Additive)
    mean = 0
    sigma = 0.05
    gaussian = np.random.normal(mean, sigma, img.shape).astype('float32')
    img = img + gaussian
    
    # 2. Salt and Pepper Noise
    s_vs_p = 0.5
    amount = 0.005 # Percentage of pixels to affect
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    img[tuple(coords)] = 1.0 # White (Salt)
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    img[tuple(coords)] = 0.0 # Black (Pepper)
    
    return np.clip(img, 0.0, 1.0) # Clip values to [0, 1]

# Custom Dataset class
class DirtyDocumentsDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256), split='train', add_noise=False, noise_type='mixed'):
        self.data_dir = os.path.join(root_dir, split)
        self.target_size = target_size
        self.add_noise = add_noise
        self.noise_type = noise_type
        # Ensure the directory exists before listing files
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}. Check your 'dirty_root' path.")
            
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Loaded {len(self.image_files)} clean images from {self.data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Load and Preprocess CLEAN image
        clean_img = cv2.imread(img_path)
        if clean_img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        clean_img = cv2.resize(clean_img, self.target_size, interpolation=cv2.INTER_AREA)
        clean_img = clean_img.astype(np.float32) / 255.0 # Normalize to [0, 1]
        
        # Create NOISY image
        if self.add_noise and self.noise_type == 'mixed':
            noisy_img = add_mixed_noise(clean_img)
        else:
            noisy_img = clean_img.copy() # If add_noise is False, noisy = clean
        
        # Convert to Tensors (PyTorch expects [C, H, W])
        clean_tensor = torch.from_numpy(clean_img.transpose(2, 0, 1)).float()
        noisy_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1)).float()
        
        # The main training loop in difussion_gpy.py doesn't use the output of this dataloader, 
        # but we return the pairs for completeness.
        return noisy_tensor, clean_tensor

# The required function
def create_dirty_documents_dataloader(root_dir, batch_size, target_size, split='train', add_noise=False, noise_type='mixed', num_workers=0):
    """Factory function to create a DataLoader for the Dirty Documents Dataset."""
    dataset = DirtyDocumentsDataset(
        root_dir=root_dir,
        target_size=target_size,
        split=split,
        add_noise=add_noise,
        noise_type=noise_type
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle for training
        num_workers=num_workers
    )
    return dataloader