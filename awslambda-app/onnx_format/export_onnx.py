import torch
import os
import boto3
import timm

def download_latest_model():
    """Download the latest model from S3"""
    try:
        # Create S3 client using environment variables
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        
        # Download the model
        s3.download_file(
            'model-deployment-data',
            'latest/model.pt',
            'model/latest_model.pt'
        )
        print("Successfully downloaded latest model")

    except Exception as e:
        raise Exception(f"Error downloading model: {str(e)}")
        
def export_model_to_onnx(
    model_path="model/latest_model.pt", 
    output_path="model/onnx_model.onnx",
    num_classes=13
):
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Loading model...")
    # Create model architecture
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Remove 'model.' prefix if present
    if all(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)  # Changed size to 224x224

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}},
        opset_version=11
    )
    print(f"Model exported successfully to {output_path}")

if __name__ == "__main__":
    download_latest_model() 
    export_model_to_onnx()