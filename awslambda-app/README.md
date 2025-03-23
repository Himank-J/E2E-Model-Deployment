# ML Model Inferencing with AWS Lambda, AWS CDK and FastAPI
[![Deploy to AWS Lambda](https://github.com/Himank-J/Lamda-deploy-cdk/actions/workflows/cdk-deploy.yml/badge.svg)](https://github.com/Himank-J/Lamda-deploy-cdk/actions/workflows/cdk-deploy.yml)

This repository demonstrates the deployment of a classification model for inference using AWS Lambda, AWS CDK, and FastAPI, with a frontend powered by FastHTML.

## End Product 

Click to view the [Web App](https://pm4xtg27s23utbddy2elaqjafi0onric.lambda-url.ap-south-1.on.aws/)

<img width="1241" alt="image" src="https://github.com/user-attachments/assets/bbf4032b-fd1a-4b3a-a0e3-9d51c676ff67" />

## Key Highlights

**1. AWS Lambda**

AWS Lambda is utilized to run the classification inference function in a serverless environment. This ensures scalability, cost-efficiency, and minimal management overhead by automatically provisioning resources based on incoming requests.

**2. AWS CDK**

The AWS Cloud Development Kit (CDK) simplifies infrastructure provisioning for this project. It is used to:

- Define and deploy Lambda functions.
- Set up API Gateway for routing HTTP requests to the Lambda function.
- Configure necessary IAM roles and policies. This IaC approach allows rapid iteration, better version control, and repeatability in deployments.

**3. FastAPI and ONNX Model**

FastAPI serves as the inference API, offering high-performance endpoints for interacting with the classification model.
ONNX (Open Neural Network Exchange Format) is chosen for the model due to its:
- Interoperability across multiple platforms.
- Optimized runtime performance, particularly in resource-constrained environments like AWS Lambda.
- Reduced inference latency and size compared to native model formats.
  
**4. FastHTML for Frontend**

The lightweight FastHTML framework is employed to create a user-friendly interface. This allows users to:

- Upload files for classification.
- View results in real-time. The simplicity of FastHTML complements the serverless architecture by reducing frontend complexity.

**5. CI/CD using Github Actions**

We also utilise GHA for pushing changes to our Lambda App. using CI/CD any time you make any change, it will trigger a pipeline that will deploy the changes on AWS Lambda. The same can be used to destroy all infra resources without having to manually remove/delete them

-----

## Implementation Details

**Dockerfile**
```python
FROM public.ecr.aws/docker/library/python:3.12-slim

# Copy the Lambda adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Set environment variables
ENV PORT=8000

# Set working directory
WORKDIR /var/task

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app.py ./
COPY icons.py ./
COPY models/ ./models/

# Set command
CMD exec uvicorn --host 0.0.0.0 --port $PORT app:app 
```

**cdk.py**
```python
class DogClassifierFastAPIStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create Lambda function
        lambda_fn = DockerImageFunction(
            self,
            "DogClassifierFastAPI",
            code=DockerImageCode.from_image_asset(
                str(Path.cwd()),  # Uses the Dockerfile in the current directory
                file="Dockerfile",
            ),
            architecture=Architecture.X86_64,
            memory_size=1092,  # 1GB memory
            timeout=Duration.minutes(5),
        )

        # Add HTTPS URL
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)

        # Output the function URL
        CfnOutput(
            self,
            "FunctionUrl",
            value=fn_url.url,
            description="URL for the Lambda function",
        )

app = App()
DogClassifierFastAPIStack(app, "DogClassifierFastAPIStack", env=cdk_environment)
app.synth()
```

- The `DogClassifierFastAPIStack` class is what creates the Lambda App.

```python
lambda_fn = DockerImageFunction(
    self,
    "DogClassifierFastAPI",
    code=DockerImageCode.from_image_asset(
        str(Path.cwd()),  # Uses the Dockerfile in the current directory
        file="Dockerfile",
    ),
    architecture=Architecture.X86_64,
    memory_size=1092,  # 1GB memory
    timeout=Duration.minutes(5),
)
```
We use above code for using custom Dockerfile on AWS Lambda along with specification for memory resource and timeout

**CI/CD**
```bash
name: Deploy to AWS Lambda

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ secrets.AWS_REGION }}
    
    - name: Install dependencies
      run: |
        npm install -g aws-cdk
    
    - name: Synthesize CDK Stack
      run: cdk synth
    
    - name: CDK Deploy
      run: cdk deploy --require-approval never
```
- We first install python and nodejs dependencies
- Then we configure our AWS credentials
- We install aws-cdk cli using `npm install -g aws-cdk` and deploy using `cdk deploy`

---

## Conclusion

This streamlined architecture demonstrates a robust, scalable solution for deploying ML models in production using modern serverless technologies.

---

## References

1. [AWS CDK](https://aws.amazon.com/cdk/)
2. [AWS Lambda](https://aws.amazon.com/lambda/)
3. [FastHTML](https://fastht.ml/)

