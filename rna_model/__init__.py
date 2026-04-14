"""RNA 3D Folding Pipeline - Modular Architecture"""

# Core pipeline and utilities
from .core import (
    RNAFoldingPipeline, PipelineConfig, GlobalConfig, get_config, validate_config,
    setup_logging, StructuredLogger, compute_tm_score, compute_rmsd, superimpose_coordinates,
    compute_contact_map, bin_distances, mask_sequence, set_seed, clear_cache, memory_usage,
    GeometryModule, RigidTransform, RNASampler, SamplerConfig, PerformanceMetrics,
    GeometryRefiner
)

# Neural network models
from .models import (
    RNALanguageModel, masked_span_loss, contact_loss,
    SecondaryStructurePredictor, secondary_structure_loss,
    StructureEncoder, IntegratedModel, Trainer, TrainingConfig,
    StructureEvaluator, EvaluationMetrics
)

# Data handling
from .data import DatasetManager, RNAStructure

# Analysis and optimization tools
from .core.analysis import ResultAnalyzer, AnalysisConfig, analyze_experiment_results, compare_experiment_results, analyze_dataset_performance, generate_analysis_report
from .core.experiment import ExperimentManager, ExperimentConfig, ExperimentResults, create_experiment_config, log_training_results
from .core.optimization import HyperparameterTuner, HyperparameterSpace, quick_hyperparameter_search, comprehensive_hyperparameter_search

__version__ = "0.1.0"
__all__ = [
    # Core pipeline
    "RNAFoldingPipeline",
    "PipelineConfig",
    "GlobalConfig",
    "get_config",
    "validate_config",
    "setup_logging",
    "StructuredLogger",
    
    # Models
    "RNALanguageModel",
    "SecondaryStructurePredictor",
    "StructureEncoder",
    "IntegratedModel",
    "GeometryModule",
    "RNASampler",
    "SamplerConfig",
    "PerformanceMetrics",
    "GeometryRefiner",
    "Trainer",
    "TrainingConfig",
    "StructureEvaluator",
    "EvaluationMetrics",
    
    # Data
    "DatasetManager",
    "RNAStructure",
    
    # Utilities
    "compute_tm_score",
    "compute_rmsd",
    "superimpose_coordinates",
    "compute_contact_map",
    "bin_distances",
    "mask_sequence",
    "set_seed",
    "clear_cache",
    "memory_usage",
    "RigidTransform",
    
    # Loss functions
    "masked_span_loss",
    "contact_loss",
    "secondary_structure_loss",
    
    # Analysis and optimization
    "ExperimentManager",
    "ExperimentConfig",
    "ExperimentResults",
    "create_experiment_config",
    "log_training_results",
    "HyperparameterTuner",
    "HyperparameterSpace",
    "quick_hyperparameter_search",
    "comprehensive_hyperparameter_search",
    "ResultAnalyzer",
    "AnalysisConfig",
    "analyze_experiment_results",
    "compare_experiment_results",
    "analyze_dataset_performance",
    "generate_analysis_report",
]
