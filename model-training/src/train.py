import os
import sys
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import glob
import json
from datetime import datetime

# Find project root from .project-root file
def get_project_root():
    """Get project root path from .project-root file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != '/':
        if os.path.exists(os.path.join(os.path.dirname(current_dir), '.project-root')):
            return os.path.dirname(current_dir)  # Return parent of model-training
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Could not find .project-root file")

# Set PROJECT_ROOT environment variable
os.environ["PROJECT_ROOT"] = get_project_root()

# Add model-training to PYTHONPATH
sys.path.append(os.path.join(os.environ["PROJECT_ROOT"], "model-training"))

from src.data.datamodule import ImageClassificationDataModule
from src.models.timm_module import TIMMLightningModule

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Set random seed
    seed_everything(cfg.seed)
    
    # Create output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    os.makedirs(cfg.paths.models_dir, exist_ok=True)  
    
    # Create model_output directory
    model_output_dir = os.path.join(cfg.paths.model_dir, 'results')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize data module
    datamodule = ImageClassificationDataModule(
        data_dir=cfg.dataset.path,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers
    )
    
    # Check for existing checkpoint
    checkpoint_path = find_latest_checkpoint(cfg.paths.checkpoint_dir)
    
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        # Load model from checkpoint
        model = TIMMLightningModule.load_from_checkpoint(checkpoint_path)
    else:
        print("Starting new training")
        # Initialize new model
        model = hydra.utils.instantiate(cfg.model)
    
    # Initialize callbacks
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))
    
    # Initialize trainer
    trainer = hydra.utils.instantiate(cfg.training, callbacks=callbacks)
    
    # Train model
    trainer.fit(model, datamodule=datamodule)
    
    # Test model
    test_results = trainer.test(model, datamodule=datamodule)
    
    # Prepare results dictionary
    results = {
        "dataset": cfg.dataset.name,
        "model": cfg.model.model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "test_results": test_results[0],  # test_results is a list with one dict
        "training_config": {
            "max_epochs": cfg.training.max_epochs,
            "batch_size": cfg.dataset.batch_size,
            "learning_rate": cfg.model.learning_rate,
            "weight_decay": cfg.model.weight_decay,
            "num_classes": cfg.dataset.num_classes
        }
    }
    
    # Save test results
    results_filename = f"{cfg.dataset.name}_{cfg.model.model_name}_results.json"
    results_path = os.path.join(cfg.paths.results_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Test results saved to: {results_path}")
    
    # Save final model to models directory
    final_model_path = os.path.join(
        cfg.paths.models_dir, 
        f"{cfg.dataset.name}_{cfg.model.model_name}_final.ckpt"
    )
    trainer.save_checkpoint(final_model_path)
    print(f"Model saved to: {final_model_path}")

if __name__ == "__main__":
    main() 