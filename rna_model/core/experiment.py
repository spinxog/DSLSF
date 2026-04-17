"""Experiment management for RNA 3D folding research."""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_name: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResults:
    """Results from an experiment."""
    experiment_id: str
    config_hash: str
    training_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    model_path: str
    checkpoint_path: str
    training_time: float
    timestamp: str
    status: str = "completed"
    notes: str = ""


class ExperimentManager:
    """Manage experiments for RNA 3D folding research."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        self.experiments_file = self.experiments_dir / "experiments.json"
        self.experiments_registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry."""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save experiment registry."""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments_registry, f, indent=2)
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment."""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.get_hash()}"
        
        # Create experiment directory
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Update registry
        self.experiments_registry[experiment_id] = {
            'id': experiment_id,
            'config_hash': config.get_hash(),
            'config_path': str(config_path),
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'experiment_name': config.experiment_name,
            'description': config.description,
            'tags': config.tags
        }
        
        self._save_registry()
        
        print(f"Created experiment: {experiment_id}")
        return experiment_id
    
    def log_results(self, experiment_id: str, results: ExperimentResults):
        """Log results for an experiment."""
        if experiment_id not in self.experiments_registry:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Update experiment directory
        exp_dir = self.experiments_dir / experiment_id
        
        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        
        # Update registry
        self.experiments_registry[experiment_id].update({
            'status': results.status,
            'completed_at': results.timestamp,
            'training_time': results.training_time,
            'results_path': str(results_path),
            'model_path': results.model_path,
            'checkpoint_path': results.checkpoint_path,
            'notes': results.notes
        })
        
        self._save_registry()
        
        print(f"Logged results for experiment: {experiment_id}")
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment details."""
        if experiment_id not in self.experiments_registry:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp_info = self.experiments_registry[experiment_id].copy()
        exp_dir = self.experiments_dir / experiment_id
        
        # Load configuration
        if (exp_dir / "config.json").exists():
            with open(exp_dir / "config.json", 'r') as f:
                exp_info['config'] = json.load(f)
        
        # Load results if available
        if (exp_dir / "results.json").exists():
            with open(exp_dir / "results.json", 'r') as f:
                exp_info['results'] = json.load(f)
        
        return exp_info
    
    def list_experiments(self, status: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List experiments with optional filtering."""
        experiments = []
        
        for exp_id, exp_info in self.experiments_registry.items():
            # Filter by status
            if status and exp_info.get('status') != status:
                continue
            
            # Filter by tags
            if tags and not any(tag in exp_info.get('tags', []) for tag in tags):
                continue
            
            experiments.append(exp_info)
        
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        comparison = {
            'experiments': [],
            'metrics_comparison': {},
            'config_comparison': {}
        }
        
        for exp_id in experiment_ids:
            exp_info = self.get_experiment(exp_id)
            comparison['experiments'].append(exp_info)
        
        if len(comparison['experiments']) > 1:
            # Compare metrics
            all_metrics = set()
            for exp in comparison['experiments']:
                if 'results' in exp and 'validation_metrics' in exp['results']:
                    all_metrics.update(exp['results']['validation_metrics'].keys())
            
            for metric in all_metrics:
                metric_values = {}
                for exp in comparison['experiments']:
                    if 'results' in exp and 'validation_metrics' in exp['results']:
                        metric_values[exp['id']] = exp['results']['validation_metrics'].get(metric)
                comparison['metrics_comparison'][metric] = metric_values
        
        return comparison
    
    def get_best_experiment(self, metric: str, higher_is_better: bool = True) -> Optional[str]:
        """Get best experiment based on a metric."""
        best_exp_id = None
        best_value = None
        
        for exp_id, exp_info in self.experiments_registry.items():
            if exp_info.get('status') != 'completed':
                continue
            
            exp_dir = self.experiments_dir / exp_id
            if not (exp_dir / "results.json").exists():
                continue
            
            with open(exp_dir / "results.json", 'r') as f:
                results = json.load(f)
            
            if 'validation_metrics' in results and metric in results['validation_metrics']:
                value = results['validation_metrics'][metric]
                
                if best_value is None:
                    best_value = value
                    best_exp_id = exp_id
                elif higher_is_better and value > best_value:
                    best_value = value
                    best_exp_id = exp_id
                elif not higher_is_better and value < best_value:
                    best_value = value
                    best_exp_id = exp_id
        
        return best_exp_id
    
    def export_results(self, output_path: str, format: str = "json"):
        """Export all experiment results."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.experiments_registry),
            'experiments': []
        }
        
        for exp_id in self.experiments_registry:
            exp_info = self.get_experiment(exp_id)
            export_data['experiments'].append(exp_info)
        
        output_file = Path(output_path)
        if format == "json":
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == "csv":
            import pandas as pd
            df_data = []
            for exp in export_data['experiments']:
                if 'results' in exp and 'validation_metrics' in exp['results']:
                    row = {
                        'experiment_id': exp['id'],
                        'experiment_name': exp.get('experiment_name', ''),
                        'status': exp.get('status', ''),
                        'created_at': exp.get('created_at', ''),
                        'completed_at': exp.get('completed_at', ''),
                        'training_time': exp.get('training_time', 0),
                        'notes': exp.get('notes', '')
                    }
                    row.update(exp['results']['validation_metrics'])
                    df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_file, index=False)
        
        print(f"Exported {len(export_data['experiments'])} experiments to {output_path}")


# Convenience functions for common research workflows
def create_experiment_config(name: str, model_config: Dict, training_config: Dict, 
                          dataset_config: Dict, description: str = "", tags: List[str] = None) -> ExperimentConfig:
    """Create experiment configuration."""
    return ExperimentConfig(
        experiment_name=name,
        model_config=model_config,
        training_config=training_config,
        dataset_config=dataset_config,
        description=description,
        tags=tags or []
    )


def log_training_results(experiment_id: str, training_metrics: Dict, validation_metrics: Dict,
                        model_path: str, checkpoint_path: str, training_time: float,
                        notes: str = ""):
    """Log training results for an experiment."""
    manager = ExperimentManager()
    
    results = ExperimentResults(
        experiment_id=experiment_id,
        config_hash="",  # Will be filled by manager
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        training_time=training_time,
        timestamp=datetime.now().isoformat(),
        notes=notes
    )
    
    manager.log_results(experiment_id, results)
    return results
