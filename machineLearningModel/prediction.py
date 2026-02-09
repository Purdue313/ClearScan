import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np

# =========================
# DEVICE SETUP (FIXED)
# =========================

def get_device():
    """Get the best available device with proper handling"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Try Intel GPU with DirectML
    try:
        import torch_directml
        device = torch_directml.device()
        print("Using Intel GPU (DirectML)")
        return device
    except:
        pass
    
    # Fall back to CPU
    print("Using CPU")
    return torch.device("cpu")

DEVICE = get_device()

# =========================
# CONFIG
# =========================

CHECKPOINT_PATH = str(Path(__file__).resolve().parent / "chexpert_epoch_2.pth")

LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

# =========================
# LOAD MODEL (FIXED)
# =========================

def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    
    # FIXED: Load to CPU first, then move to device
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Now move to the target device
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"  Training loss: {checkpoint['train_loss']:.4f}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model

# =========================
# IMAGE PREPROCESSING
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# PREDICTION FUNCTION
# =========================

def predict_xray(model, image_path):
    """
    Runs inference and returns ranked predictions.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

    ranked = sorted(
        zip(LABELS, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "label": label,
            "probability": float(prob),
            "percentage": f"{prob * 100:.2f}%"
        }
        for label, prob in ranked
    ]

# =========================
# GRAD-CAM IMPLEMENTATION (INLINE)
# =========================

import torch.nn.functional as F
import cv2

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Generates heatmaps showing which regions of an X-ray the model
    focuses on when making predictions.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Your trained ResNet model
            target_layer: Layer to extract gradients from (e.g., model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for a specific class
        
        Args:
            input_tensor: Preprocessed image tensor [1, 3, 224, 224]
            class_idx: Target class index (if None, uses highest prediction)
        
        Returns:
            heatmap: Numpy array [H, W] with values 0-1
        """
        self.model.eval()
        
        # Get the device from input tensor
        device = input_tensor.device
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If class_idx not specified, use the highest prediction
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero all gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = output[0, class_idx]
        target.backward()
        
        # Get activations and gradients (they're already on the correct device)
        activations = self.activations[0]  # [C, H, W]
        gradients = self.gradients[0]      # [C, H, W]
        
        # Global average pooling of gradients (importance weights)
        weights = torch.mean(gradients, dim=[1, 2])  # [C]
        
        # Weighted combination of activation maps - ensure on same device
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Convert to numpy
        heatmap = cam.cpu().numpy()
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap [H, W]
            original_image: PIL Image or numpy array
            alpha: Transparency (0=only image, 1=only heatmap)
            colormap: OpenCV colormap
        
        Returns:
            overlay: PIL Image with heatmap overlay
        """
        # Convert PIL to numpy if needed
        if isinstance(original_image, Image.Image):
            img = np.array(original_image.convert('RGB'))
        else:
            img = original_image.copy()
        
        # Resize heatmap to match image
        h, w = img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        
        return Image.fromarray(overlay)
    
    def overlay_with_contours(self, heatmap, original_image, threshold=0.5, alpha=0.4):
        """
        Overlay heatmap with contour lines like medical imaging style
        
        Args:
            heatmap: Grad-CAM heatmap [H, W]
            original_image: PIL Image
            threshold: Only show regions above this activation
            alpha: Transparency
        
        Returns:
            overlay: PIL Image with contoured heatmap
        """
        # Convert image
        img = np.array(original_image.convert('RGB'))
        h, w = img.shape[:2]
        
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Create mask for high-activation regions
        mask = (heatmap_resized > threshold).astype(np.uint8) * 255
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Only show heatmap where activation is high
        for c in range(3):
            heatmap_colored[:, :, c] = np.where(
                mask > 0,
                heatmap_colored[:, :, c],
                0
            )
        
        # Blend
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Find and draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), 2)
        
        return Image.fromarray(overlay)

# =========================
# HEATMAP GENERATION FUNCTIONS
# =========================

def predict_with_heatmap(model, image_path, target_label=None, save_path=None):
    """
    Generate prediction with Grad-CAM heatmap overlay
    
    Args:
        model: Trained model
        image_path: Path to X-ray image
        target_label: Specific condition to visualize (or None for top prediction)
        save_path: Where to save the heatmap (optional)
    
    Returns:
        Dictionary with predictions and heatmap image (PIL Image)
    """
    # Get predictions
    predictions = predict_xray(model, image_path)
    
    # Load original image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, model.layer4)
    
    # Determine target class
    if target_label is None:
        # Use top prediction
        label = predictions[0]['label']
    else:
        label = target_label
    
    # Get the class index
    class_idx = LABELS.index(label)
    
    # Generate heatmap
    heatmap = gradcam.generate_heatmap(input_tensor, class_idx=class_idx)
    
    # Create overlay with contours (medical imaging style)
    overlay = gradcam.overlay_with_contours(
        heatmap,
        image,
        threshold=0.5,
        alpha=0.4
    )
    
    # Save if requested
    if save_path:
        overlay.save(save_path)
        print(f"Heatmap saved to: {save_path}")
    
    return {
        'predictions': predictions,
        'heatmap_image': overlay,
        'target_condition': label,
        'probability': next(p['probability'] for p in predictions if p['label'] == label)
    }

# =========================
# BATCH PREDICTION
# =========================

def predict_batch(model, image_folder, output_csv="predictions.csv", threshold=0.5):
    """
    Predict on multiple X-rays in a folder
    
    Args:
        model: Trained model
        image_folder: Folder containing X-ray images
        output_csv: Where to save results
        threshold: Probability threshold
    """
    import os
    import pandas as pd
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    print(f"Found {len(image_files)} images in {image_folder}")
    
    # Predict on each image
    all_results = []
    
    for image_path in image_files:
        print(f"Processing: {os.path.basename(image_path)}")
        results = predict_xray(model, image_path)
        
        # Prepare row for CSV
        row = {'Image': os.path.basename(image_path)}
        for r in results:
            row[r['label']] = r['probability']
        all_results.append(row)
        
        # Print top predictions
        print("  Top predictions:")
        for i, r in enumerate(results[:3]):
            print(f"    {i+1}. {r['label']}: {r['percentage']}")
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    return df