"""Hyperparameter optimization for RNA 3D folding research."""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import itertools
import random
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from .experiment import ExperimentManager, ExperimentConfig, ExperimentResults
from ..training import Trainer, TrainingConfig
from .pipeline import RNAFoldingPipeline, PipelineConfig


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space."""
    
    # Model hyperparameters
    d_model: Tuple[int, int] = (256, 1024)  # (min, max)
    n_layers: Tuple[int, int] = (4, 16)
    n_heads: Tuple[int, int] = (4, 16)
    d_ff: Tuple[int, int] = (512, 4096)
    
    # Training hyperparameters
    learning_rate: Tuple[float, float] = (1e-5, 1e-3)
    weight_decay: Tuple[float, float] = (1e-6, 1e-4)
    batch_size: Tuple[int, int] = (4, 32)
    warmup_steps: Tuple[int, int] = (100, 2000)
    
    # Sampler hyperparameters
    temperature: Tuple[float, float] = (0.5, 2.0)
    n_decoys: Tuple[int, int] = (3, 10)
    n_steps: Tuple[int, int] = (500, 2000)
    min_distance: Tuple[float, float] = (2.0, 5.0)
    
    # Geometry hyperparameters
    contact_threshold: Tuple[float, float] = (6.0, 12.0)
    rmsd_threshold: Tuple[float, float] = (3.0, 8.0)


class HyperparameterOptimizer:
    """Base class for hyperparameter optimization."""
    
    def __init__(self, space: HyperparameterSpace, 
                 objective_metric: str = "tm_score",
                 maximize: bool = True,
                 max_evaluations: int = 100):
        self.space = space
        self.objective_metric = objective_metric
        self.maximize = maximize
        self.max_evaluations = max_evaluations
        self.evaluation_history = []
        
    def sample_hyperparameters(self) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}
        
        # Model hyperparameters
        params['d_model'] = random.randint(*self.space.d_model)
        params['n_layers'] = random.randint(*self.space.n_layers)
        params['n_heads'] = random.randint(*self.space.n_heads)
        params['d_ff'] = random.randint(*self.space.d_ff)
        
        # Training hyperparameters
        params['learning_rate'] = 10 ** random.uniform(np.log10(self.space.learning_rate[0]), 
                                                    np.log10(self.space.learning_rate[1]))
        params['weight_decay'] = 10 ** random.uniform(np.log10(self.space.weight_decay[0]), 
                                                      np.log10(self.space.weight_decay[1]))
        params['batch_size'] = random.choice([2**i for i in range(2, 6) 
                                               if 2**i >= self.space.batch_size[0] and 2**i <= self.space.batch_size[1]])
        params['warmup_steps'] = random.randint(*self.space.warmup_steps)
        
        # Sampler hyperparameters
        params['temperature'] = random.uniform(*self.space.temperature)
        params['n_decoys'] = random.randint(*self.space.n_decoys)
        params['n_steps'] = random.randint(*self.space.n_steps)
        params['min_distance'] = random.uniform(*self.space.min_distance)
        
        # Geometry hyperparameters
        params['contact_threshold'] = random.uniform(*self.space.contact_threshold)
        params['rmsd_threshold'] = random.uniform(*self.space.rmsd_threshold)
        
        return params
    
    def evaluate_hyperparameters(self, params: Dict[str, Any], 
                              train_data: List, val_data: List) -> float:
        """Evaluate hyperparameters (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def optimize(self, train_data: List, val_data: List) -> Dict[str, Any]:
        """Run optimization (to be implemented by subclasses)."""
        raise NotImplementedError


class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search hyperparameter optimization."""
    
    def __init__(self, space: HyperparameterSpace, **kwargs):
        super().__init__(space, **kwargs)
        self.grid_points = self._generate_grid()
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate grid of hyperparameter combinations."""
        grid = []
        
        # Define grid points for each parameter
        d_model_values = [256, 512, 1024]
        n_layers_values = [4, 8, 12, 16]
        n_heads_values = [4, 8, 16]
        learning_rate_values = [1e-5, 1e-4, 1e-3]
        batch_size_values = [8, 16, 32]
        temperature_values = [0.5, 1.0, 1.5, 2.0]
        n_decoys_values = [3, 5, 7, 10]
        
        # Generate all combinations (limit to avoid explosion)
        combinations = list(itertools.product(
            d_model_values[:2], n_layers_values[:2], n_heads_values[:2],
            learning_rate_values[:2], batch_size_values[:2],
            temperature_values[:2], n_decoys_values[:2]
        ))
        
        for combo in combinations:
            params = {
                'd_model': combo[0],
                'n_layers': combo[1],
                'n_heads': combo[2],
                'learning_rate': combo[3],
                'batch_size': combo[4],
                'temperature': combo[5],
                'n_decoys': combo[6],
                # Set reasonable defaults for other parameters
                'd_ff': combo[0] * 4,
                'weight_decay': 1e-5,
                'warmup_steps': 1000,
                'n_steps': 1000,
                'min_distance': 3.0,
                'contact_threshold': 8.0,
                'rmsd_threshold': 5.0
            }
            grid.append(params)
        
        # Limit number of evaluations
        return grid[:self.max_evaluations]
    
    def optimize(self, train_data: List, val_data: List) -> Dict[str, Any]:
        """Run grid search optimization."""
        logging.info(f"Starting grid search with {len(self.grid_points)} combinations")
        
        best_params = None
        best_score = -np.inf if self.maximize else np.inf
        all_results = []
        
        for i, params in enumerate(self.grid_points):
            logging.info(f"Evaluating combination {i+1}/{len(self.grid_points)}: {params}")
            
            try:
                score = self.evaluate_hyperparameters(params, train_data, val_data)
                
                result = {
                    'params': params,
                    'score': score,
                    'iteration': i
                }
                all_results.append(result)
                
                # Update best
                if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()
                
                logging.info(f"Score: {score:.6f} (Best: {best_score:.6f})")
                
            except Exception as e:
                logging.error(f"Error evaluating params {params}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'optimization_method': 'grid_search',
            'total_evaluations': len(self.grid_points)
        }


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search hyperparameter optimization."""
    
    def optimize(self, train_data: List, val_data: List) -> Dict[str, Any]:
        """Run random search optimization."""
        logging.info(f"Starting random search with {self.max_evaluations} evaluations")
        
        best_params = None
        best_score = -np.inf if self.maximize else np.inf
        all_results = []
        
        for i in range(self.max_evaluations):
            params = self.sample_hyperparameters()
            logging.info(f"Evaluating iteration {i+1}/{self.max_evaluations}: {params}")
            
            try:
                score = self.evaluate_hyperparameters(params, train_data, val_data)
                
                result = {
                    'params': params,
                    'score': score,
                    'iteration': i
                }
                all_results.append(result)
                
                # Update best
                if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()
                
                logging.info(f"Score: {score:.6f} (Best: {best_score:.6f})")
                
            except Exception as e:
                logging.error(f"Error evaluating params {params}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'optimization_method': 'random_search',
            'total_evaluations': self.max_evaluations
        }


class BayesianOptimizer(HyperparameterOptimizer):
    """Simple Bayesian optimization using Gaussian processes."""
    
    def __init__(self, space: HyperparameterSpace, **kwargs):
        super().__init__(space, **kwargs)
        self.acquisition_function = 'ei'  # Expected Improvement
        self.observations = []
    
    def _acquisition_function(self, mean, std, best_y):
        """Expected Improvement acquisition function."""
        from scipy.stats import norm
        
        improvement = mean - best_y
        Z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        return ei
    
    def optimize(self, train_data: List, val_data: List) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        logging.info(f"Starting Bayesian optimization with {self.max_evaluations} evaluations")
        
        # Initial random sampling
        n_initial = min(10, self.max_evaluations // 2)
        
        best_params = None
        best_score = -np.inf if self.maximize else np.inf
        all_results = []
        
        # Initial random samples
        for i in range(n_initial):
            params = self.sample_hyperparameters()
            score = self.evaluate_hyperparameters(params, train_data, val_data)
            
            result = {
                'params': params,
                'score': score,
                'iteration': i
            }
            all_results.append(result)
            self.observations.append(result)
            
            if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                best_score = score
                best_params = params.copy()
        
        # Bayesian optimization
        for i in range(n_initial, self.max_evaluations):
            # Simple surrogate model (mean of nearby points)
            params = self._suggest_next_point()
            score = self.evaluate_hyperparameters(params, train_data, val_data)
            
            result = {
                'params': params,
                'score': score,
                'iteration': i
            }
            all_results.append(result)
            self.observations.append(result)
            
            if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                best_score = score
                best_params = params.copy()
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'optimization_method': 'bayesian_optimization',
            'total_evaluations': self.max_evaluations
        }
    
    def _suggest_next_point(self) -> Dict[str, Any]:
        """Suggest next point to evaluate (simplified)."""
        # For simplicity, use random sampling around best points
        if len(self.observations) == 0:
            return self.sample_hyperparameters()
        
        # Find best observation
        best_obs = max(self.observations, key=lambda x: x['score'])
        best_params = best_obs['params']
        
        # Sample around best point with some noise
        params = best_params.copy()
        
        # Add noise to continuous parameters
        if 'learning_rate' in params:
            params['learning_rate'] *= np.random.uniform(0.8, 1.2)
        if 'temperature' in params:
            params['temperature'] *= np.random.uniform(0.8, 1.2)
        if 'min_distance' in params:
            params['min_distance'] *= np.random.uniform(0.8, 1.2)
        
        # Randomly modify discrete parameters occasionally
        if random.random() < 0.3:
            if 'n_decoys' in params:
                params['n_decoys'] = random.randint(*self.space.n_decoys)
            if 'batch_size' in params:
                params['batch_size'] = random.choice([8, 16, 32])
        
        return params


class HyperparameterTuner:
    """Main interface for hyperparameter tuning."""
    
    def __init__(self, space: HyperparameterSpace):
        self.space = space
        self.experiment_manager = ExperimentManager()
    
    def tune(self, train_dataset_id: str, val_dataset_id: str, 
              optimizer_type: str = 'random_search',
              max_evaluations: int = 50,
              objective_metric: str = 'tm_score',
              maximize: bool = True,
              parallel: bool = False,
              n_workers: int = 1) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        
        # Load datasets
        dataset_manager = DatasetManager()
        train_data, _ = dataset_manager.load_dataset(train_dataset_id)
        val_data, _ = dataset_manager.load_dataset(val_dataset_id)
        
        # Create optimizer
        if optimizer_type == 'grid_search':
            optimizer = GridSearchOptimizer(self.space, max_evaluations=max_evaluations, 
                                         objective_metric=objective_metric, maximize=maximize)
        elif optimizer_type == 'random_search':
            optimizer = RandomSearchOptimizer(self.space, max_evaluations=max_evaluations, 
                                            objective_metric=objective_metric, maximize=maximize)
        elif optimizer_type == 'bayesian':
            optimizer = BayesianOptimizer(self.space, max_evaluations=max_evaluations, 
                                         objective_metric=objective_metric, maximize=maximize)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Run optimization
        if parallel:
            results = self._parallel_optimize(optimizer, train_data, val_data, n_workers)
        else:
            results = optimizer.optimize(train_data, val_data)
        
        # Log optimization as experiment
        config = ExperimentConfig(
            experiment_name=f"hyperparameter_tuning_{optimizer_type}",
            model_config=results['best_params'],
            training_config={'max_evaluations': max_evaluations},
            dataset_config={'train_dataset': train_dataset_id, 'val_dataset': val_dataset_id},
            description=f"Hyperparameter tuning using {optimizer_type}",
            tags=[optimizer_type, 'hyperparameter_tuning']
        )
        
        experiment_id = self.experiment_manager.create_experiment(config)
        
        # Create results object
        experiment_results = ExperimentResults(
            experiment_id=experiment_id,
            config_hash=config.get_hash(),
            training_metrics={'optimization_method': optimizer_type},
            validation_metrics={objective_metric: results['best_score']},
            model_path="",  # Will be filled by training
            checkpoint_path="",
            training_time=0.0,
            timestamp=datetime.now().isoformat(),
            notes=f"Best score: {results['best_score']:.6f}"
        )
        
        self.experiment_manager.log_results(experiment_id, experiment_results)
        
        # Add detailed results to experiment
        results['experiment_id'] = experiment_id
        results['objective_metric'] = objective_metric
        results['maximize'] = maximize
        
        return results
    
    def _parallel_optimize(self, optimizer: HyperparameterOptimizer, 
                         train_data: List, val_data: List, n_workers: int) -> Dict[str, Any]:
        """Run optimization in parallel."""
        all_results = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all evaluation tasks
            futures = []
            for i in range(optimizer.max_evaluations):
                params = optimizer.sample_hyperparameters()
                future = executor.submit(optimizer.evaluate_hyperparameters, params, train_data, val_data)
                futures.append((future, params, i))
            
            # Collect results
            for future, params, i in futures:
                try:
                    score = future.result()
                    result = {
                        'params': params,
                        'score': score,
                        'iteration': i
                    }
                    all_results.append(result)
                    
                    # Update optimizer state
                    optimizer.evaluation_history.append(result)
                    
                except Exception as e:
                    logging.error(f"Error in parallel evaluation {i}: {e}")
        
        # Find best result
        best_result = max(all_results, key=lambda x: x['score']) if optimizer.maximize else min(all_results, key=lambda x: x['score'])
        
        return {
            'best_params': best_result['params'],
            'best_score': best_result['score'],
            'all_results': all_results,
            'optimization_method': 'parallel_' + optimizer.__class__.__name__.lower(),
            'total_evaluations': len(all_results)
        }


# Convenience functions for common research workflows
def quick_hyperparameter_search(train_dataset_id: str, val_dataset_id: str, 
                              max_evaluations: int = 20) -> Dict[str, Any]:
    """Quick hyperparameter search with default settings."""
    space = HyperparameterSpace()
    tuner = HyperparameterTuner(space)
    
    return tuner.tune(
        train_dataset_id=train_dataset_id,
        val_dataset_id=val_dataset_id,
        optimizer_type='random_search',
        max_evaluations=max_evaluations,
        objective_metric='tm_score',
        maximize=True
    )


def comprehensive_hyperparameter_search(train_dataset_id: str, val_dataset_id: str,
                                     max_evaluations: int = 100) -> Dict[str, Any]:
    """Comprehensive hyperparameter search with multiple optimizers."""
    space = HyperparameterSpace()
    tuner = HyperparameterTuner(space)
    
    results = {}
    for optimizer_type in ['random_search', 'grid_search']:
        try:
            result = tuner.tune(
                train_dataset_id=train_dataset_id,
                val_dataset_id=val_dataset_id,
                optimizer_type=optimizer_type,
                max_evaluations=max_evaluations // 2,
                objective_metric='tm_score',
                maximize=True
            )
            results[optimizer_type] = result
            
            print(f"{optimizer_type} - Best score: {result['best_score']:.6f}")
            
        except Exception as e:
            logging.error(f"Error in {optimizer_type}: {e}")
            results[optimizer_type] = {'error': str(e)}
    
    # Find overall best
    best_overall = None
    best_score = -np.inf
    
    for optimizer_type, result in results.items():
        if 'error' not in result and result['best_score'] > best_score:
            best_score = result['best_score']
            best_overall = optimizer_type
    
    results['overall_best'] = {
        'optimizer': best_overall,
        'best_score': best_score,
        'all_results': results
    }
    
    return results
