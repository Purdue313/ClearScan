import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

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
        print("✓ Using Intel GPU (DirectML)")
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
    
    print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs)")
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
        results = predict_xray(model, image_path, threshold)
        
        # Prepare row for CSV
        row = {'Image': os.path.basename(image_path)}
        row.update(results['predictions'])
        all_results.append(row)
        
        # Print detected abnormalities
        if results['detected_abnormalities']:
            print("  Detected:")
            for det in results['detected_abnormalities']:
                print(f"    - {det['condition']}: {det['probability']:.1%}")
        else:
            print("  No abnormalities detected")
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    return df

# =========================
# EXAMPLE USAGE
# =========================

if __name__ == "__main__":
    import os
    
    # Load your trained model
    model = load_model(CHECKPOINT_PATH)
    
    print("\n" + "="*60)
    print("CHEST X-RAY ABNORMALITY DETECTOR")
    print("="*60)
    print("1. Single image prediction")
    print("2. Batch prediction (folder of images)")
    print("="*60)
    
    mode = input("Enter choice (1 or 2): ").strip()
    
    if mode == "1":
        # Single image prediction
        image_path = input("Enter path to X-ray image: ").strip().strip('"')
        
        if os.path.exists(image_path):
            print("\nAnalyzing X-ray...")
            results = predict_xray(model, image_path)
            
            print("\n" + "="*60)
            print(f"Results for: {os.path.basename(image_path)}")
            print("="*60)
            
            # Sort by probability
            sorted_preds = sorted(results['predictions'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            print("\nAll predictions:")
            for label, prob in sorted_preds:
                bar = "█" * int(prob * 20)
                print(f"  {label:30s} {prob:6.1%} {bar}")
            
            if results['detected_abnormalities']:
                print("\n🔴 Detected abnormalities (>50% probability):")
                for det in results['detected_abnormalities']:
                    print(f"  • {det['condition']}: {det['probability']:.1%}")
            else:
                print("\n✅ No significant abnormalities detected")
        else:
            print(f"Error: Image not found at {image_path}")
    
    elif mode == "2":
        # Batch prediction
        folder_path = input("Enter path to folder with X-rays: ").strip().strip('"')
        output_csv = input("Enter output CSV filename (default: predictions.csv): ").strip()
        
        if not output_csv:
            output_csv = "predictions.csv"
        
        if os.path.exists(folder_path):
            print("\nProcessing batch...")
            df = predict_batch(model, folder_path, output_csv)
            
            print("\n" + "="*60)
            print("Batch processing complete!")
            print("="*60)
            print(f"Processed {len(df)} images")
            print(f"Results saved to: {output_csv}")
        else:
            print(f"Error: Folder not found at {folder_path}")
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("⚠️  DISCLAIMER:")
    print("This is an AI prediction tool, not a medical diagnosis.")
    print("Always consult qualified healthcare professionals.")
    print("="*60)