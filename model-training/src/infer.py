import os
import sys
import hydra
from omegaconf import DictConfig
import torch
from PIL import Image
from torchvision import transforms
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# Find project root from .project-root file
def get_project_root():
    """Get project root path from .project-root file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != '/':
        project_root_file = os.path.join(current_dir, '.project-root')
        if os.path.exists(project_root_file):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Could not find .project-root file")

# Set PROJECT_ROOT environment variable
os.environ["PROJECT_ROOT"] = get_project_root()

# Add project root to PYTHONPATH
sys.path.append(os.environ["PROJECT_ROOT"])

from src.models.timm_module import TIMMLightningModule

def load_model(model_path):
    """Load the trained model"""
    model = TIMMLightningModule.load_from_checkpoint(model_path)
    model.eval()
    return model

def get_transforms(image_size):
    """Get transforms for inference"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def process_image(image_path, model, transform, class_names):
    """Process a single image and return predictions"""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return image, class_names[predicted.item()], confidence.item()

def save_prediction(image, class_name, confidence, output_path):
    """Save prediction using matplotlib"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    
    # Add text with prediction
    plt.title(f"{class_name}: {confidence:.2%}", 
              pad=20,
              fontsize=14,
              bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2, dpi=150)
    plt.close()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Create input/output directories
    input_dir = cfg.paths.infer_input_dir
    output_dir = cfg.paths.infer_output_dir
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_path = os.path.join(
        cfg.paths.models_dir,
        f"{cfg.dataset.name}_{cfg.model.model_name}_final.ckpt"
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = load_model(model_path)
    transform = get_transforms(cfg.dataset.image_size)
    
    # Get class names from the training data directory
    class_names = sorted(os.listdir(os.path.join(cfg.dataset.path, 'train')))
    
    # Process all images in input directory
    image_files = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
                 glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))
    
    print(f"Found {len(image_files)} images to process")
    
    for image_path in image_files:
        print(f"Processing {image_path}")
        
        # Process image
        image, predicted_class, confidence = process_image(
            image_path, model, transform, class_names
        )
        
        # Save prediction
        output_path = os.path.join(
            output_dir, 
            f"pred_{Path(image_path).stem}.jpg"
        )
        save_prediction(image, predicted_class, confidence, output_path)
        
        print(f"Saved prediction to {output_path}")
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main() 