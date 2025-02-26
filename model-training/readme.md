export PROJECT_ROOT=$(pwd)

# Initialize DVC    
dvc init

# Add your S3 remote (replace with your bucket details)
dvc remote add -d myremote s3://your-bucket-name

# Configure AWS credentials if needed
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

dvc add data/intel
dvc add data/sports
git add data/*.dvc
git commit -m "Add data tracking with DVC"