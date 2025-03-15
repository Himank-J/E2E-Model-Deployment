import gradio as gr
import torch
from utils.model_loader import ModelLoader

# Initialize model loader
model_loader = ModelLoader(bucket_name="model-deployment-data")
model = model_loader.model
transform = model_loader.get_transforms()
labels = model_loader.labels
facts = model_loader.facts

def predict(image):
    """Make prediction and return label, confidence, and fact"""
    if image is None:
        return None, None
        
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Create prediction dictionary for all classes
    predictions = {
        labels[idx]: float(prob)
        for idx, prob in enumerate(probabilities[0])
    }
    
    # Get the fact for the top prediction
    top_label = max(predictions.items(), key=lambda x: x[1])[0]
    fact = facts[top_label]
    
    return predictions, fact

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Classification Results"),
        gr.Textbox(label="Fun Fact About This Category!")
    ],
    title="🎯 Scene and Sport Classification",
    description="""
    ## Classify Scenes and Sports!
    Upload a clear photo, and I'll classify it into one of our categories and share an interesting fact about it! 
    This model can identify various scenes and sports activities with high accuracy.
    
    ### Supported Categories:
    **Scenes**: Buildings, Forest, Glacier, Mountain, Sea, Street
    **Sports**: Badminton, Baseball, Basketball, Football, Rowing, Swimming, Tennis
    """,
    article="""
    ### Tips for best results:
    - Use clear, well-lit photos
    - Ensure the main subject is visible
    - Avoid blurry or dark images
    
    ### Model Information:
    - This model is automatically updated with the best performing version through our CI/CD pipeline
    - Latest model accuracy and performance metrics are tracked and monitored
    - Trained on a combined dataset of natural scenes and sports activities
    """,
    examples=[
        ["examples/basketball.png"],
        ["examples/boxing.png"],
        ["examples/buildings.png"],
        ["examples/cricket.png"],
        ["examples/football.png"],
        ["examples/forest.png"],
        ["examples/formula_racing.png"],
        ["examples/glacier.png"],
        ["examples/golf.png"],
        ["examples/hockey.png"],
        ["examples/mountain.png"],
        ["examples/sea.png"],
        ["examples/street.png"]
    ],
    theme=gr.themes.Citrus(),
    css="footer {display: none !important;}"
)

if __name__ == "__main__":
    iface.launch() 