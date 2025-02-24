import os
import sys
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import glob

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
    trainer.test(model, datamodule=datamodule)
    
    # Save final model to model-data directory
    final_model_path = os.path.join(
        cfg.paths.model_dir, 
        f"{cfg.dataset.name}_{cfg.model.model_name}_final.ckpt"
    )
    trainer.save_checkpoint(final_model_path)
    print(f"Model saved to: {final_model_path}")

if __name__ == "__main__":
    main() 