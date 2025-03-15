import boto3
import torch
from PIL import Image
import os
import json
from torchvision import transforms
import timm
from pathlib import Path

class ModelLoader:
    def __init__(self, bucket_name: str, model_name: str = "resnet18", num_classes: int = 13):
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories if they don't exist
        os.makedirs('model', exist_ok=True)
        
        # Set AWS credentials path
        self.aws_credentials_path = self.get_aws_credentials_path()
        
        # Download and load model
        self.download_latest_model()
        self.model = self.load_model()
        
        # Load labels and facts
        self.labels = self.get_labels()
        self.facts = self.get_facts()
        
    def get_aws_credentials_path(self):
        """Get the path to AWS credentials file"""
        # Check current directory first
        local_aws_dir = Path('.aws')
        if local_aws_dir.exists():
            return local_aws_dir
        
        # Check home directory next
        home_aws_dir = Path.home() / '.aws'
        if home_aws_dir.exists():
            return home_aws_dir
        
        raise FileNotFoundError("AWS credentials directory not found")
        
    def download_latest_model(self):
        """Download the latest model from S3"""
        try:
            # Create boto3 session with specific credentials file
            session = boto3.Session(
                profile_name='default',
                shared_credentials_file=str(self.aws_credentials_path / 'credentials')
            )
            
            # Create S3 client using the session
            s3 = session.client('s3')
            
            # Download the model
            s3.download_file(
                self.bucket_name,
                'latest/model.pt',
                'model/latest_model.pt'
            )
            print("Successfully downloaded latest model")
        except Exception as e:
            raise Exception(f"Error downloading model: {str(e)}")
    
    def load_model(self):
        """Load the model with weights"""
        model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.num_classes
        )
        model.load_state_dict(torch.load('model/latest_model.pt', map_location=self.device))
        model.eval()
        return model
    
    def get_labels(self):
        """Get class labels"""
        return [
            "basketball", "boxing", "buildings", "cricket", 
            "football", "forest", "formula 1 racing", "glacier",
            "golf", "hockey", "mountain", "sea", "street"
        ]
    
    def get_facts(self):
        """Get interesting facts about each class"""
        return {
            "basketball": "The NBA's three-point line is 23'9\" from the basket.",
            "boxing": "The first Olympic boxing competition was held in 1904.",
            "buildings": "Modern skyscrapers use advanced materials and engineering to reach incredible heights.",
            "cricket": "Cricket is the second most popular sport in the world.",
            "football": "A soccer ball must be between 27-28 inches in circumference.",
            "forest": "Forests cover about 31% of the world's land surface and are crucial for biodiversity.",
            "formula 1 racing": "The fastest lap in a Formula 1 race is 1:15.328.",
            "glacier": "Glaciers store about 69% of the world's fresh water.",
            "golf": "Golf was first played in Scotland in the 15th century.",
            "hockey": "Hockey is the national sport of Canada.",
            "mountain": "Mount Everest grows about 4mm higher every year.",
            "sea": "The ocean contains 97% of Earth's water and covers 71% of the planet's surface.",
            "street": "The oldest known paved road was built in Egypt around 2600 BC."
        }
    
    def get_transforms(self):
        """Get image transforms for inference"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) 