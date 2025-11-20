# scripts/extract_metrics.py
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup Databricks MLflow connection
from config.mlflow_config import setup_mlflow
setup_mlflow()

# Now your existing code
import mlflow
from tabulate import tabulate

# Get all runs from experiment
experiment_id = "4464541790971986"
runs = mlflow.search_runs(experiment_id)

print(f"\nFound {len(runs)} runs\n")

# Create comparison table
comparison = []
for _, run in runs.iterrows():
    # Get metrics (they're already percentages, not decimals)
    best_val_acc = run.get('metrics.best_val_acc', None)
    final_val_acc = run.get('metrics.val_acc', None)
    
    comparison.append({
        'Run Name': run.get('tags.mlflow.runName', 'N/A'),
        'Model': run.get('params.model_name', 'N/A'),
        'Batch Size': run.get('params.batch_size', 'N/A'),
        'LR': run.get('params.learning_rate', 'N/A'),
        'Best Val Acc': f"{best_val_acc:.2f}%" if best_val_acc else 'N/A',
        'Final Val Acc': f"{final_val_acc:.2f}%" if final_val_acc else 'N/A',
    })

# Sort by best val acc (descending)
comparison_sorted = sorted(comparison, 
                          key=lambda x: float(x['Best Val Acc'].replace('%', '').replace('N/A', '0')), 
                          reverse=True)

print("="*100)
print("EXPERIMENT RESULTS - Sorted by Best Validation Accuracy")
print("="*100)
print(tabulate(comparison_sorted, headers='keys', tablefmt='grid'))
print("="*100)