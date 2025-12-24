import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- 1. THE NOISE GENERATOR (For datasets like Tobacco800) ---
def add_synthetic_noise(image):
    """
    Takes a clean image and adds random noise (Gaussian + Salt & Pepper).
    """
    img_noisy = image.copy()
    
    # Randomly choose noise intensity
    sigma = np.random.uniform(0.02, 0.1)  # Variable static
    salt_pepper_ratio = np.random.uniform(0.01, 0.05) # Variable dots
    
    # A. Gaussian Noise (Static)
    gaussian = np.random.normal(0, sigma, img_noisy.shape).astype('float32')
    img_noisy = img_noisy + gaussian
    
    # B. Salt and Pepper (Black/White dots)
    # Salt (White)
    num_salt = np.ceil(salt_pepper_ratio * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    img_noisy[tuple(coords)] = 1.0
    
    # Pepper (Black)
    num_pepper = np.ceil(salt_pepper_ratio * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    img_noisy[tuple(coords)] = 0.0
    
    return np.clip(img_noisy, 0.0, 1.0)

# --- 2. THE DATASET CLASS ---
class UniversalDataset(Dataset):
    def __init__(self, root_dir, mode='auto', target_size=(256, 256)):
        """
        Args:
            root_dir: Path to the dataset folder.
            mode: 'paired' (looks for folders), 'single' (generates noise), or 'auto' (guesses).
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.mode = mode
        self.image_paths = []
        
        # --- A. Auto-Detect Mode ---
        if self.mode == 'auto':
            # Check if "train" and "train_cleaned" exist (Dirty Docs format)
            if os.path.exists(os.path.join(root_dir, 'train')) and \
               os.path.exists(os.path.join(root_dir, 'train_cleaned')):
                self.mode = 'paired'
                self.noisy_dir = os.path.join(root_dir, 'train')
                self.clean_dir = os.path.join(root_dir, 'train_cleaned')
                print(f"[Loader] Detected PAIRED dataset at {root_dir}")
                
                # Load file list from Noisy folder
                self.image_paths = [f for f in os.listdir(self.noisy_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            else:
                self.mode = 'single'
                print(f"[Loader] Detected SINGLE folder dataset at {root_dir}. Will generate synthetic noise.")
                
                # Load file list recursively from root
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                            self.image_paths.append(os.path.join(root, file))

                self.image_paths = self.image_paths[:500]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- CASE 1: PAIRED DATA (Dirty Docs) ---
        if self.mode == 'paired':
            img_name = self.image_paths[idx]
            
            # Load Noisy
            noisy_path = os.path.join(self.noisy_dir, img_name)
            noisy_img = self._load_and_process(noisy_path)
            
            # Load Clean
            clean_path = os.path.join(self.clean_dir, img_name)
            if os.path.exists(clean_path):
                clean_img = self._load_and_process(clean_path)
            else:
                clean_img = noisy_img # Fallback
                
        # --- CASE 2: SINGLE DATA (Tobacco800) ---
        else:
            # The file path is stored directly in image_paths for single mode
            img_path = self.image_paths[idx]
            clean_img = self._load_and_process(img_path)
            
            # Generate Noisy version on the fly
            noisy_img = add_synthetic_noise(clean_img)

        # Convert to PyTorch Tensors [Channels, Height, Width]
        noisy_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1)).float()
        clean_tensor = torch.from_numpy(clean_img.transpose(2, 0, 1)).float()
        
        return noisy_tensor, clean_tensor, "image"

    def _load_and_process(self, path):
        """Helper to load, resize, normalize, AND FIX INVERTED images"""
        img = cv2.imread(path)
        if img is None:
            # Create black placeholder if load fails
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        
        # Normalize to 0-1
        img_norm = img.astype(np.float32) / 255.0
        
        # --- NEW: SMART FLIP ---
        # Calculate the average brightness. 
        # If it's less than 0.5, the page is likely Black (Inverted).
        if np.mean(img_norm) < 0.5:
            # Invert it to make it White Background / Black Text
            img_norm = 1.0 - img_norm
            
        return img_norm

# --- 3. HELPER FUNCTION TO GET LOADER ---
def get_dataloader(root_dir, batch_size=2):
    dataset = UniversalDataset(root_dir=root_dir, mode='auto')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)