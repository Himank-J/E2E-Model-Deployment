import json
import boto3
from datetime import datetime
import os
from typing import Dict, Tuple, Optional

def get_latest_model_metrics(bucket: str) -> Optional[Tuple[Dict, str]]:
    """Get metrics from the most recent model in S3"""
    s3 = boto3.client('s3')
    
    # List all date-based directories
    response = s3.list_objects_v2(
        Bucket=bucket,
        Delimiter='/'
    )

    # Get the latest date directory
    date_prefixes = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
    if not date_prefixes:
        return None
    
    latest_date = max(date_prefixes)

    # List commit directories in the latest date
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=latest_date,
        Delimiter='/'
    )

    # Get the latest commit
    commit_prefixes = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
    if not commit_prefixes:
        return None
    
    latest_commit = commit_prefixes[-1]  # Last commit in the date directory

    # Get the metrics file
    metrics_key = f"{latest_commit}model-data/results/combined_resnet18_results.json"
    try:
        response = s3.get_object(Bucket=bucket, Key=metrics_key)
        metrics = json.loads(response['Body'].read().decode('utf-8'))
        return metrics, latest_commit
    except Exception as e:
        return None

def compare_models(current_metrics: Dict, previous_metrics: Optional[Dict]) -> Dict:
    """Compare current and previous model metrics"""
    
    if previous_metrics is None:
        # No previous model, return current metrics as baseline
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_model": {
                "commit": os.getenv("GITHUB_SHA", "unknown")[:7],
                "trained_at": current_metrics["timestamp"],
                "accuracy": current_metrics["test_results"]["test_acc"]
            },
            "previous_model": None,
            "changes": None,
            "config_changes": None
        }
    
    def format_change(current: float, previous: float) -> str:
        change = ((current - previous) / previous) * 100
        return f"{change:+.2f}%"
    
    comparison = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_model": {
            "commit": os.getenv("GITHUB_SHA", "unknown")[:7],
            "trained_at": current_metrics["timestamp"],
            "accuracy": current_metrics["test_results"]["test_acc"]
        },
        "previous_model": {
            "commit": "previous",
            "trained_at": previous_metrics["timestamp"],
            "accuracy": previous_metrics["test_results"]["test_acc"]
        },
        "changes": {
            "accuracy": format_change(
                current_metrics["test_results"]["test_acc"],
                previous_metrics["test_results"]["test_acc"]
            )
        },
        "config_changes": {}
    }
    
    # Compare training configurations
    for key in current_metrics["training_config"]:
        curr_val = current_metrics["training_config"][key]
        prev_val = previous_metrics["training_config"][key]
        if curr_val != prev_val:
            comparison["config_changes"][key] = {
                "from": prev_val,
                "to": curr_val
            }
    
    return comparison

def generate_comparison_report(comparison: Dict) -> str:
    """Generate a markdown report from the comparison"""
    
    if comparison["previous_model"] is None:
        # Generate baseline report
        report = [
            "### Model Training Report",
            "",
            f"**Commit:** {comparison['current_model']['commit']}",
            f"**Timestamp:** {comparison['current_model']['trained_at']}",
            "",
            "#### Performance Metrics",
            "```json",
            json.dumps({
                "accuracy": comparison["current_model"]["accuracy"]
            }, indent=2),
            "```",
            "",
            "#### No previous model to compare with.",
            ""
        ]
    else:
        # Generate comparison report
        report = [
            "## 🔄 Model Performance Comparison",
            "",
            "### 📊 Performance Metrics",
            "",
            "| Metric | Previous Model | Current Model | Change |",
            "|--------|---------------|---------------|--------|",
            f"| Accuracy | {comparison['previous_model']['accuracy']:.4f} | {comparison['current_model']['accuracy']:.4f} | {comparison['changes']['accuracy']} |",
            "",
        ]
        
        if comparison["config_changes"]:
            report.extend([
                "### ⚙️ Configuration Changes",
                "",
                "| Parameter | Previous Value | New Value |",
                "|-----------|----------------|-----------|"
            ])
            
            for param, change in comparison["config_changes"].items():
                report.append(f"| {param} | {change['from']} | {change['to']} |")
        
        report.extend([
            "",
            "### 📝 Details",
            "",
            f"- Previous Model Trained: {comparison['previous_model']['trained_at']}",
            f"- Current Model Trained: {comparison['current_model']['trained_at']}",
            f"- Comparison Generated: {comparison['timestamp']}"
        ])
    
    return "\n".join(report)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python model_comparison.py <s3_bucket> <current_metrics_path>")
        sys.exit(1)
    
    bucket = sys.argv[1]
    current_metrics_path = sys.argv[2]
    
    # Load current metrics
    with open(current_metrics_path) as f:
        current_metrics = json.load(f)
    
    # Get previous metrics from S3
    previous_metrics, latest_commit = get_latest_model_metrics(bucket)
    
    # Compare models
    comparison = compare_models(current_metrics, previous_metrics)
    
    # Generate report
    report = generate_comparison_report(comparison)
    
    # Save report to file
    with open('comparison_report.md', 'w') as f:
        f.write(report)
    
    print("Comparison report generated and saved as comparison_report.md")