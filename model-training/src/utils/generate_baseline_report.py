import json
import os
from datetime import datetime

def generate_training_report(metrics_path: str, infer_count: int, commit_id: str, datetime_str: str, bucket_name: str) -> str:
    """Generate a markdown report for the current model training run"""
    
    # Read metrics from JSON
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create report
    report = [
        "### Model Training Report",
        "",
        f"**Commit:** {commit_id}",
        f"**Timestamp:** {datetime_str}",
        "",
        "#### Performance Metrics",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
        "",
        "#### Validation Set Inference",
        f"- Number of samples processed: {infer_count}",
        "- Inference results stored in S3",
        "",
        "#### Storage Location",
        f"s3://{bucket_name}/{datetime_str}/{commit_id}/model-data/"
    ]
    
    return "\n".join(report)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 6:
        print("Usage: python generate_report.py <metrics_path> <infer_output_dir> <commit_id> <datetime> <bucket_name>")
        sys.exit(1)
    
    metrics_path = sys.argv[1]
    infer_output_dir = sys.argv[2]
    commit_id = sys.argv[3]
    datetime_str = sys.argv[4]
    bucket_name = sys.argv[5]
    
    # Count inference samples
    infer_count = len([f for f in os.listdir(infer_output_dir) if f.endswith('.jpg')])
    
    # Generate report
    report = generate_training_report(
        metrics_path=metrics_path,
        infer_count=infer_count,
        commit_id=commit_id,
        datetime_str=datetime_str,
        bucket_name=bucket_name
    )
    
    # Write report to file
    with open('baseline_report.md', 'w') as f:
        f.write(report) 
    
    print("Baseline report generated and saved as baseline_report.md")