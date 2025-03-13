import json
import boto3
from datetime import datetime
import os
from typing import Dict, Tuple, Optional, List

def get_two_latest_models(bucket: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Get metrics from the two most recent models in S3"""
    s3 = boto3.client('s3')
    
    # List all date-based directories
    response = s3.list_objects_v2(
        Bucket=bucket,
        Delimiter='/'
    )
    
    # Get all date prefixes and sort them
    date_prefixes = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
    if not date_prefixes:
        return None, None
    
    # Sort date prefixes in descending order (most recent first)
    date_prefixes.sort(reverse=True)
    
    # Get the two most recent dates
    latest_dates = date_prefixes[:2]
    print(f"Found date directories: {latest_dates}")  
    
    metrics_list = []
    for date_prefix in latest_dates:
        # List commit directories in each date
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=date_prefix,
            Delimiter='/'
        )
        
        # Get all commit directories for this date
        commit_prefixes = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
        if not commit_prefixes:
            continue
            
        print(f"Found commits for {date_prefix}: {commit_prefixes}") 
        
        # For each date, get the latest commit
        latest_commit = commit_prefixes[-1]
        
        # Construct the full path to metrics file
        metrics_key = f"{latest_commit}model-data/results/combined_resnet18_results.json"
        print(f"Looking for metrics at: {metrics_key}")  
        
        try:
            response = s3.get_object(Bucket=bucket, Key=metrics_key)
            metrics = json.loads(response['Body'].read().decode('utf-8'))
            # Add commit info to metrics
            metrics['commit_id'] = latest_commit.rstrip('/').split('/')[-1]
            metrics['run_date'] = date_prefix.rstrip('/')
            metrics_list.append(metrics)
            print(f"Successfully loaded metrics from {metrics_key}")  
        except Exception as e:
            print(f"Error reading metrics from {metrics_key}: {str(e)}")
            continue
    
    # Return the two most recent metrics
    if len(metrics_list) == 0:
        return None, None
    elif len(metrics_list) == 1:
        return metrics_list[0], None
    else:
        return metrics_list[0], metrics_list[1]  # current, previous

def compare_models(current_metrics: Dict, previous_metrics: Optional[Dict]) -> Dict:
    """Compare current and previous model metrics"""
    
    if previous_metrics is None:
        # No previous model, return current metrics as baseline
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_model": {
                "commit": current_metrics['commit_id'],
                "run_date": current_metrics['run_date'],
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
            "commit": current_metrics['commit_id'],
            "run_date": current_metrics['run_date'],
            "trained_at": current_metrics["timestamp"],
            "accuracy": current_metrics["test_results"]["test_acc"]
        },
        "previous_model": {
            "commit": previous_metrics['commit_id'],
            "run_date": previous_metrics['run_date'],
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
    
    if len(sys.argv) != 2:
        print("Usage: python model_comparison.py <s3_bucket>")
        sys.exit(1)
    
    bucket = sys.argv[1]
    
    # Get the two most recent models from S3
    current_metrics, previous_metrics = get_two_latest_models(bucket)
    
    if current_metrics is None:
        print("No models found in S3")
        sys.exit(1)
    
    # Compare models
    comparison = compare_models(current_metrics, previous_metrics)
    
    # Generate report
    report = generate_comparison_report(comparison)
    
    # Save report to file
    with open('comparison_report.md', 'w') as f:
        f.write(report)
    
    print("Comparison report generated and saved as comparison_report.md")