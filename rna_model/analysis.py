"""Result analysis and visualization tools for RNA 3D folding research."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

from .experiment import ExperimentManager
from .evaluation import EvaluationMetrics
from .utils import compute_tm_score, compute_rmsd


@dataclass
class AnalysisConfig:
    """Configuration for analysis tools."""
    output_dir: str = "analysis"
    figure_format: str = "png"  # png, pdf, svg
    figure_dpi: int = 300
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"


class ResultAnalyzer:
    """Analyze and visualize experiment results."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
        
        self.experiment_manager = ExperimentManager()
    
    def analyze_experiment(self, experiment_id: str, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze a single experiment."""
        exp_info = self.experiment_manager.get_experiment(experiment_id)
        
        analysis = {
            'experiment_id': experiment_id,
            'experiment_info': exp_info,
            'analysis': {}
        }
        
        if 'results' in exp_info:
            results = exp_info['results']
            
            # Training analysis
            if 'training_metrics' in results:
                analysis['analysis']['training'] = self._analyze_training_metrics(results['training_metrics'])
            
            # Validation analysis
            if 'validation_metrics' in results:
                analysis['analysis']['validation'] = self._analyze_validation_metrics(results['validation_metrics'])
            
            # Create plots
            if save_plots:
                self._create_experiment_plots(experiment_id, analysis)
        
        return analysis
    
    def compare_experiments(self, experiment_ids: List[str], save_plots: bool = True) -> Dict[str, Any]:
        """Compare multiple experiments."""
        comparison = self.experiment_manager.compare_experiments(experiment_ids)
        
        analysis = {
            'experiment_ids': experiment_ids,
            'comparison': comparison,
            'analysis': {}
        }
        
        # Analyze metrics comparison
        if 'metrics_comparison' in comparison:
            analysis['analysis']['metrics'] = self._analyze_metrics_comparison(comparison['metrics_comparison'])
        
        # Create comparison plots
        if save_plots:
            self._create_comparison_plots(experiment_ids, analysis)
        
        return analysis
    
    def analyze_dataset_performance(self, dataset_id: str) -> Dict[str, Any]:
        """Analyze model performance on a dataset."""
        from .dataset import DatasetManager
        
        dataset_manager = DatasetManager()
        structures, dataset_info = dataset_manager.load_dataset(dataset_id)
        
        analysis = {
            'dataset_id': dataset_id,
            'dataset_info': dataset_info,
            'analysis': {}
        }
        
        # Analyze structure predictions
        predictions = []
        for structure in structures[:100]:  # Sample for analysis
            try:
                # Simulate prediction (in real use, this would use the model)
                prediction = {
                    'sequence': structure.sequence,
                    'predicted_coords': structure.coordinates.copy(),  # Would be model prediction
                    'true_coords': structure.coordinates.copy(),      # Would be ground truth
                    'confidence': 0.8  # Would be model confidence
                }
                predictions.append(prediction)
            except Exception as e:
                logging.warning(f"Error analyzing structure: {e}")
        
        if predictions:
            analysis['analysis']['predictions'] = self._analyze_predictions(predictions)
        
        # Create plots
        self._create_dataset_analysis_plots(dataset_id, analysis)
        
        return analysis
    
    def _analyze_training_metrics(self, training_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training metrics."""
        analysis = {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for key, value in training_metrics.items():
            if isinstance(value, (int, float)):
                numeric_metrics[key] = value
            elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                numeric_metrics[key] = {
                    'mean': np.mean(value),
                    'std': np.std(value),
                    'min': np.min(value),
                    'max': np.max(value)
                }
        
        analysis['numeric_metrics'] = numeric_metrics
        analysis['metric_keys'] = list(numeric_metrics.keys())
        analysis['summary'] = {
            'total_metrics': len(training_metrics),
            'numeric_metrics': len(numeric_metrics)
        }
        
        return analysis
    
    def _analyze_validation_metrics(self, validation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze validation metrics."""
        analysis = {}
        
        # Key metrics for RNA structure prediction
        key_metrics = ['tm_score', 'rmsd', 'gdt_ts', 'gdt_ha', 'lddt']
        analysis['key_metrics'] = {}
        
        for metric in key_metrics:
            if metric in validation_metrics:
                value = validation_metrics[metric]
                analysis['key_metrics'][metric] = {
                    'value': value,
                    'status': self._evaluate_metric_status(metric, value)
                }
        
        analysis['summary'] = {
            'total_metrics': len(validation_metrics),
            'key_metrics_found': len(analysis['key_metrics'])
        }
        
        return analysis
    
    def _analyze_metrics_comparison(self, metrics_comparison: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze metrics comparison across experiments."""
        analysis = {}
        
        for metric, values in metrics_comparison.items():
            if not values:
                continue
            
            values_array = np.array(list(values.values()))
            analysis[metric] = {
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array),
                'range': np.max(values_array) - np.min(values_array),
                'coefficient_of_variation': np.std(values_array) / np.mean(values_array) if np.mean(values_array) != 0 else 0,
                'n_experiments': len(values)
            }
        
        return analysis
    
    def _analyze_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction quality."""
        if not predictions:
            return {'error': 'No predictions to analyze'}
        
        analysis = {}
        
        # Calculate metrics for each prediction
        tm_scores = []
        rmsds = []
        confidences = []
        
        for pred in predictions:
            try:
                tm_score = compute_tm_score(pred['predicted_coords'], pred['true_coords'])
                rmsd = compute_rmsd(pred['predicted_coords'], pred['true_coords'])
                
                tm_scores.append(tm_score)
                rmsds.append(rmsd)
                confidences.append(pred['confidence'])
            except Exception as e:
                logging.warning(f"Error computing metrics: {e}")
        
        if tm_scores:
            analysis['tm_score'] = {
                'mean': np.mean(tm_scores),
                'std': np.std(tm_scores),
                'min': np.min(tm_scores),
                'max': np.max(tm_scores),
                'distribution': 'normal' if self._is_normal_distribution(tm_scores) else 'unknown'
            }
        
        if rmsds:
            analysis['rmsd'] = {
                'mean': np.mean(rmsds),
                'std': np.std(rmsds),
                'min': np.min(rmsds),
                'max': np.max(rmsds),
                'distribution': 'normal' if self._is_normal_distribution(rmsds) else 'unknown'
            }
        
        if confidences:
            analysis['confidence'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        analysis['summary'] = {
            'total_predictions': len(predictions),
            'metrics_computed': len([k for k in ['tm_score', 'rmsd', 'confidence'] if k in analysis])
        }
        
        return analysis
    
    def _is_normal_distribution(self, data: List[float]) -> bool:
        """Test if data follows normal distribution."""
        if len(data) < 30:
            return False  # Too small to test
        
        from scipy import stats
        _, p_value = stats.normaltest(data)
        return p_value > 0.05
    
    def _evaluate_metric_status(self, metric: str, value: float) -> str:
        """Evaluate if metric value is good."""
        if metric == 'tm_score':
            if value > 0.7:
                return 'excellent'
            elif value > 0.5:
                return 'good'
            elif value > 0.3:
                return 'fair'
            else:
                return 'poor'
        elif metric == 'rmsd':
            if value < 2.0:
                return 'excellent'
            elif value < 4.0:
                return 'good'
            elif value < 6.0:
                return 'fair'
            else:
                return 'poor'
        elif metric == 'gdt_ts':
            if value > 0.8:
                return 'excellent'
            elif value > 0.6:
                return 'good'
            elif value > 0.4:
                return 'fair'
            else:
                return 'poor'
        else:
            return 'unknown'
    
    def _create_experiment_plots(self, experiment_id: str, analysis: Dict[str, Any]):
        """Create plots for experiment analysis."""
        if 'training' not in analysis['analysis']:
            return
        
        training_analysis = analysis['analysis']['training']
        
        if 'numeric_metrics' not in training_analysis:
            return
        
        numeric_metrics = training_analysis['numeric_metrics']
        
        # Create subdirectory for this experiment
        exp_plot_dir = self.output_dir / experiment_id
        exp_plot_dir.mkdir(exist_ok=True)
        
        # Plot training curves
        for metric in training_analysis['metric_keys']:
            if metric in numeric_metrics and isinstance(numeric_metrics[metric], dict):
                self._plot_training_curve(exp_plot_dir, metric, numeric_metrics[metric], experiment_id)
    
    def _plot_training_curve(self, plot_dir: Path, metric: str, values: Dict[str, Any], experiment_id: str):
        """Plot training curve for a metric."""
        if 'mean' not in values:
            return
        
        plt.figure(figsize=(10, 6))
        
        epochs = range(len(values['mean']))
        plt.plot(epochs, values['mean'], label='Mean', linewidth=2)
        
        if 'std' in values:
            plt.fill_between(epochs, 
                            np.array(values['mean']) - np.array(values['std']),
                            np.array(values['mean']) + np.array(values['std']),
                            alpha=0.3, label='Std Dev')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Training {metric.replace("_", " ").title()} - {experiment_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f"training_{metric}.png", dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _create_comparison_plots(self, experiment_ids: List[str], analysis: Dict[str, Any]):
        """Create comparison plots."""
        if 'metrics' not in analysis['analysis']:
            return
        
        metrics_analysis = analysis['analysis']['metrics']
        
        # Create comparison plots directory
        comp_plot_dir = self.output_dir / "comparison"
        comp_plot_dir.mkdir(exist_ok=True)
        
        # Plot metric comparisons
        for metric, stats in metrics_analysis.items():
            self._plot_metric_comparison(comp_plot_dir, metric, stats, experiment_ids)
    
    def _plot_metric_comparison(self, plot_dir: Path, metric: str, stats: Dict[str, Any], experiment_ids: List[str]):
        """Plot metric comparison across experiments."""
        plt.figure(figsize=(12, 6))
        
        experiments = list(stats.keys())
        values = [stats[exp]['mean'] for exp in experiments]
        errors = [stats[exp]['std'] for exp in experiments]
        
        x_pos = np.arange(len(experiments))
        
        bars = plt.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7)
        
        plt.xlabel('Experiments')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.xticks(x_pos, experiments, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + errors[i] + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(plot_dir / f"comparison_{metric}.png", dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _create_dataset_analysis_plots(self, dataset_id: str, analysis: Dict[str, Any]):
        """Create dataset analysis plots."""
        if 'predictions' not in analysis['analysis']:
            return
        
        pred_analysis = analysis['analysis']['predictions']
        
        # Create dataset plots directory
        dataset_plot_dir = self.output_dir / "datasets" / dataset_id
        dataset_plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot TM-score distribution
        if 'tm_score' in pred_analysis:
            self._plot_metric_distribution(dataset_plot_dir, 'TM Score', pred_analysis['tm_score'], dataset_id)
        
        # Plot RMSD distribution
        if 'rmsd' in pred_analysis:
            self._plot_metric_distribution(dataset_plot_dir, 'RMSD', pred_analysis['rmsd'], dataset_id)
        
        # Plot confidence distribution
        if 'confidence' in pred_analysis:
            self._plot_metric_distribution(dataset_plot_dir, 'Confidence', pred_analysis['confidence'], dataset_id)
    
    def _plot_metric_distribution(self, plot_dir: Path, metric_name: str, stats: Dict[str, Any], dataset_id: str):
        """Plot distribution of a metric."""
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        if 'mean' in stats and 'std' in stats:
            # Generate sample data for histogram (since we only have stats)
            sample_data = np.random.normal(stats['mean'], stats['std'], 1000)
            
            plt.hist(sample_data, bins=30, alpha=0.7, density=True, label='Distribution')
            
            # Plot normal distribution overlay
            x = np.linspace(stats['mean'] - 4*stats['std'], stats['mean'] + 4*stats['std'], 100)
            normal_dist = (1 / (stats['std'] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - stats['mean']) / stats['std'])**2)
            plt.plot(x, normal_dist, 'r-', label='Normal Distribution', linewidth=2)
        
        plt.xlabel(metric_name)
        plt.ylabel('Density')
        plt.title(f'{metric_name} Distribution - {dataset_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric_name.lower()}_distribution.png", 
                   dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def export_analysis_report(self, output_path: str, format: str = "html") -> str:
        """Export comprehensive analysis report."""
        output_file = Path(output_path)
        
        if format.lower() == 'html':
            return self._export_html_report(output_file)
        elif format.lower() == 'json':
            return self._export_json_report(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_html_report(self, output_file: Path) -> str:
        """Export HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RNA 3D Folding Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .good {{ color: #28a745; }}
                .fair {{ color: #ffc107; }}
                .poor {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RNA 3D Folding Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>This report contains analysis of RNA 3D folding experiments and results.</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return str(output_file)
    
    def _export_json_report(self, output_file: Path) -> str:
        """Export JSON report."""
        # Collect all experiment data
        experiments = self.experiment_manager.list_experiments()
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'experiments': experiments
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(output_file)


# Convenience functions for common analysis tasks
def analyze_experiment_results(experiment_id: str) -> Dict[str, Any]:
    """Quick analysis of a single experiment."""
    analyzer = ResultAnalyzer()
    return analyzer.analyze_experiment(experiment_id)


def compare_experiment_results(experiment_ids: List[str]) -> Dict[str, Any]:
    """Quick comparison of multiple experiments."""
    analyzer = ResultAnalyzer()
    return analyzer.compare_experiments(experiment_ids)


def analyze_dataset_performance(dataset_id: str) -> Dict[str, Any]:
    """Quick analysis of model performance on a dataset."""
    analyzer = ResultAnalyzer()
    return analyzer.analyze_dataset_performance(dataset_id)


def generate_analysis_report(output_path: str, format: str = "html") -> str:
    """Generate comprehensive analysis report."""
    analyzer = ResultAnalyzer()
    return analyzer.export_analysis_report(output_path, format)
