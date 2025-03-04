import os
import sys
import hydra
from omegaconf import DictConfig
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import glob
from pathlib import Path
import json

# Find project root from .project-root file
def get_project_root():
    """Get project root path from .project-root file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != '/':
        if os.path.exists(os.path.join(os.path.dirname(current_dir), '.project-root')):
            return os.path.dirname(current_dir)
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Could not find .project-root file")

# Set PROJECT_ROOT environment variable
os.environ["PROJECT_ROOT"] = get_project_root()
sys.path.append(os.path.join(os.environ["PROJECT_ROOT"], "model-training"))

from src.models.timm_module import TIMMLightningModule
from src.data.datamodule import ImageClassificationDataModule

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
    """Save image with prediction using matplotlib"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f'Prediction: {class_name}\nConfidence: {confidence:.2%}', 
              bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_validation_set(cfg, model, transform):
    """Process all images in the validation set"""
    # Initialize data module to get class names
    datamodule = ImageClassificationDataModule(
        data_dir=cfg.dataset.path,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers
    )
    datamodule.setup()
    class_names = datamodule.train_dataset.classes
    
    # Get validation set directory
    valid_dir = os.path.join(cfg.dataset.path, 'valid')
    results = []
    
    # Process each class directory
    for class_name in os.listdir(valid_dir):
        class_dir = os.path.join(valid_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Process each image in class directory
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, img_name)
            image, predicted_class, confidence = process_image(
                img_path, model, transform, class_names
            )
            
            # Save prediction
            output_name = f"pred_{class_name}_{Path(img_name).stem}.jpg"
            output_path = os.path.join(cfg.paths.infer_output_dir, output_name)
            save_prediction(image, predicted_class, confidence, output_path)
            
            # Store result
            results.append({
                'image': img_name,
                'true_class': class_name,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
    
    # Save results summary
    summary_path = os.path.join(cfg.paths.infer_output_dir, 'validation_results.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Create output directories
    os.makedirs(cfg.paths.infer_output_dir, exist_ok=True)
    
    # Load model
    model_path = os.path.join(
        cfg.paths.models_dir,
        f"{cfg.dataset.name}_{cfg.model.model_name}_final.ckpt"
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = load_model(model_path)
    transform = get_transforms(cfg.dataset.image_size)
    
    # Check if --valid-set-only flag is present
    if '--valid-set-only' in sys.argv:
        results = process_validation_set(cfg, model, transform)
        print(f"Processed {len(results)} validation set images")
        return
    
    # Regular inference on input directory
    input_dir = cfg.paths.infer_input_dir
    os.makedirs(input_dir, exist_ok=True)
    
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
            cfg.paths.infer_output_dir, 
            f"pred_{Path(image_path).stem}.jpg"
        )
        save_prediction(image, predicted_class, confidence, output_path)
        
        print(f"Saved prediction to {output_path}")
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main() 