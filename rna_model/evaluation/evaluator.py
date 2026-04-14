"""Evaluation utilities for RNA 3D folding pipeline."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from ..core.utils import compute_tm_score, compute_rmsd, superimpose_coordinates
from ..data import RNAStructure


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    tm_score: float
    rmsd: float
    gdt_ts: float  # Global Distance Test - Total Score
    gdt_ha: float  # Global Distance Test - High Accuracy
    lddt: float    # Local Distance Difference Test
    molprobity_clashscore: float
    best_of_5_tm: float
    median_tm: float
    confidence_correlation: float


class StructureEvaluator:
    """Evaluator for RNA structure predictions."""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_single_prediction(self,
                                  pred_coords: np.ndarray,
                                  true_coords: np.ndarray,
                                  pred_confidence: Optional[float] = None) -> Dict[str, float]:
        """Evaluate a single structure prediction."""
        # Ensure coordinates have same shape
        assert pred_coords.shape == true_coords.shape, "Coordinate shapes must match"
        
        # Superimpose structures
        aligned_pred, aligned_true = superimpose_coordinates(pred_coords, true_coords)
        
        # Compute metrics
        metrics = {}
        
        # TM-score
        metrics["tm_score"] = compute_tm_score(aligned_pred, aligned_true)
        
        # RMSD
        metrics["rmsd"] = compute_rmsd(aligned_pred, aligned_true)
        
        # GDT-TS (Global Distance Test - Total Score)
        metrics["gdt_ts"] = self._compute_gdt_ts(aligned_pred, aligned_true)
        
        # GDT-HA (Global Distance Test - High Accuracy)
        metrics["gdt_ha"] = self._compute_gdt_ha(aligned_pred, aligned_true)
        
        # LDDT (Local Distance Difference Test)
        metrics["lddt"] = self._compute_lddt(aligned_pred, aligned_true)
        
        # MolProbity clash score
        metrics["molprobity_clashscore"] = self._compute_clashscore(aligned_pred)
        
        return metrics
    
    def evaluate_ensemble(self,
                          pred_decoys: List[np.ndarray],
                          true_coords: np.ndarray,
                          confidences: Optional[List[float]] = None) -> EvaluationMetrics:
        """Evaluate ensemble of predictions (best-of-5)."""
        if not pred_decoys:
            raise ValueError("No predictions provided")
        
        # Evaluate each decoy
        decoy_metrics = []
        for i, decoy in enumerate(pred_decoys):
            confidence = confidences[i] if confidences else None
            metrics = self.evaluate_single_prediction(decoy, true_coords, confidence)
            decoy_metrics.append(metrics)
        
        # Extract individual metrics
        tm_scores = [m["tm_score"] for m in decoy_metrics]
        rmsds = [m["rmsd"] for m in decoy_metrics]
        gdt_ts_scores = [m["gdt_ts"] for m in decoy_metrics]
        gdt_ha_scores = [m["gdt_ha"] for m in decoy_metrics]
        lddt_scores = [m["lddt"] for m in decoy_metrics]
        clash_scores = [m["molprobity_clashscore"] for m in decoy_metrics]
        
        # Best-of-5 metrics
        best_idx = np.argmax(tm_scores)
        best_metrics = decoy_metrics[best_idx]
        
        # Confidence correlation (if confidences provided)
        confidence_correlation = 0.0
        if confidences:
            confidence_correlation = np.corrcoef(confidences, tm_scores)[0, 1]
            if np.isnan(confidence_correlation):
                confidence_correlation = 0.0
        
        return EvaluationMetrics(
            tm_score=best_metrics["tm_score"],
            rmsd=best_metrics["rmsd"],
            gdt_ts=best_metrics["gdt_ts"],
            gdt_ha=best_metrics["gdt_ha"],
            lddt=best_metrics["lddt"],
            molprobity_clashscore=best_metrics["molprobity_clashscore"],
            best_of_5_tm=max(tm_scores),
            median_tm=np.median(tm_scores),
            confidence_correlation=confidence_correlation
        )
    
    def _compute_gdt_ts(self, pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
        """Compute GDT-TS score."""
        thresholds = [1.0, 2.0, 4.0, 8.0]  # Å
        scores = []
        
        for threshold in thresholds:
            score = self._compute_gdt_score(pred_coords, true_coords, threshold)
            scores.append(score)
        
        return np.mean(scores)
    
    def _compute_gdt_ha(self, pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
        """Compute GDT-HA score (high accuracy)."""
        thresholds = [0.5, 1.0, 2.0, 4.0]  # Å
        scores = []
        
        for threshold in thresholds:
            score = self._compute_gdt_score(pred_coords, true_coords, threshold)
            scores.append(score)
        
        return np.mean(scores)
    
    def _compute_gdt_score(self, pred_coords: np.ndarray, true_coords: np.ndarray, threshold: float) -> float:
        """Compute GDT score for given threshold."""
        n_atoms = len(pred_coords)
        distances = np.linalg.norm(pred_coords - true_coords, axis=1)
        
        # Count atoms within threshold
        within_threshold = np.sum(distances <= threshold)
        
        return within_threshold / n_atoms
    
    def _compute_lddt(self, pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
        """Compute LDDT score."""
        n_atoms = len(pred_coords)
        lddt_scores = []
        
        for i in range(n_atoms):
            # Find neighbors within 15 Å
            distances = np.linalg.norm(true_coords - true_coords[i], axis=1)
            neighbors = np.where((distances > 0) & (distances <= 15.0))[0]
            
            if len(neighbors) == 0:
                continue
            
            # Compute distance differences
            pred_distances = np.linalg.norm(pred_coords[neighbors] - pred_coords[i], axis=1)
            true_distances = distances[neighbors]
            
            # LDDT scoring
            score = 0
            for pred_dist, true_dist in zip(pred_distances, true_distances):
                diff = abs(pred_dist - true_dist)
                if diff < 0.5:
                    score += 1
                elif diff < 1.0:
                    score += 0.5
                elif diff < 2.0:
                    score += 0.25
                # else: score += 0
            
            lddt_scores.append(score / len(neighbors))
        
        return np.mean(lddt_scores) if lddt_scores else 0.0
    
    def _compute_clashscore(self, coords: np.ndarray, threshold: float = 2.0) -> float:
        """Compute MolProbity-style clash score."""
        n_atoms = len(coords)
        clashes = 0
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < threshold:
                    clashes += 1
        
        # Clashscore = clashes / (1000 atoms)
        return (clashes / n_atoms) * 1000
    
    def evaluate_dataset(self,
                       predictions: List[List[np.ndarray]],
                       true_structures: List[RNAStructure],
                       confidences: Optional[List[List[float]]] = None) -> Dict[str, float]:
        """Evaluate predictions on entire dataset."""
        if len(predictions) != len(true_structures):
            raise ValueError("Number of predictions and true structures must match")
        
        all_metrics = []
        
        for i, (pred_decoys, true_structure) in enumerate(zip(predictions, true_structures)):
            true_coords = true_structure.coordinates[:, 0, :]  # Use P atoms
            decoy_confidences = confidences[i] if confidences else None
            
            metrics = self.evaluate_ensemble(pred_decoys, true_coords, decoy_confidences)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {}
        
        # Mean metrics
        aggregated["mean_tm_score"] = np.mean([m.tm_score for m in all_metrics])
        aggregated["mean_rmsd"] = np.mean([m.rmsd for m in all_metrics])
        aggregated["mean_gdt_ts"] = np.mean([m.gdt_ts for m in all_metrics])
        aggregated["mean_gdt_ha"] = np.mean([m.gdt_ha for m in all_metrics])
        aggregated["mean_lddt"] = np.mean([m.lddt for m in all_metrics])
        aggregated["mean_clashscore"] = np.mean([m.molprobity_clashscore for m in all_metrics])
        aggregated["mean_best_of_5_tm"] = np.mean([m.best_of_5_tm for m in all_metrics])
        aggregated["mean_median_tm"] = np.mean([m.median_tm for m in all_metrics])
        aggregated["mean_confidence_correlation"] = np.mean([m.confidence_correlation for m in all_metrics])
        
        # Standard deviations
        aggregated["std_tm_score"] = np.std([m.tm_score for m in all_metrics])
        aggregated["std_rmsd"] = np.std([m.rmsd for m in all_metrics])
        aggregated["std_best_of_5_tm"] = np.std([m.best_of_5_tm for m in all_metrics])
        
        # Success rates
        aggregated["success_rate_tm_gt_0.5"] = np.mean([m.tm_score > 0.5 for m in all_metrics])
        aggregated["success_rate_tm_gt_0.7"] = np.mean([m.tm_score > 0.7 for m in all_metrics])
        
        return aggregated
    
    def create_evaluation_report(self,
                                predictions: List[List[np.ndarray]],
                                true_structures: List[RNAStructure],
                                output_file: str,
                                confidences: Optional[List[List[float]]] = None):
        """Create detailed evaluation report."""
        # Evaluate dataset
        aggregated_metrics = self.evaluate_dataset(predictions, true_structures, confidences)
        
        # Individual metrics
        individual_metrics = []
        for i, (pred_decoys, true_structure) in enumerate(zip(predictions, true_structures)):
            true_coords = true_structure.coordinates[:, 0, :]
            decoy_confidences = confidences[i] if confidences else None
            
            metrics = self.evaluate_ensemble(pred_decoys, true_coords, decoy_confidences)
            
            individual_metrics.append({
                "sequence_id": i,
                "sequence_length": len(true_structure.sequence),
                "pdb_id": true_structure.pdb_id,
                "tm_score": metrics.tm_score,
                "rmsd": metrics.rmsd,
                "gdt_ts": metrics.gdt_ts,
                "gdt_ha": metrics.gdt_ha,
                "lddt": metrics.lddt,
                "clashscore": metrics.molprobity_clashscore,
                "best_of_5_tm": metrics.best_of_5_tm,
                "median_tm": metrics.median_tm,
                "confidence_correlation": metrics.confidence_correlation
            })
        
        # Create report
        report = {
            "summary": aggregated_metrics,
            "individual_results": individual_metrics,
            "evaluation_metadata": {
                "n_sequences": len(predictions),
                "n_decoys_per_sequence": len(predictions[0]) if predictions else 0,
                "metrics_computed": ["tm_score", "rmsd", "gdt_ts", "gdt_ha", "lddt", "clashscore"]
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {output_file}")
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Sequences evaluated: {report['evaluation_metadata']['n_sequences']}")
        print(f"Mean TM-score: {aggregated_metrics['mean_tm_score']:.3f} ± {aggregated_metrics['std_tm_score']:.3f}")
        print(f"Mean RMSD: {aggregated_metrics['mean_rmsd']:.2f} ± {aggregated_metrics['std_rmsd']:.2f} Å")
        print(f"Mean Best-of-5 TM: {aggregated_metrics['mean_best_of_5_tm']:.3f}")
        print(f"Success rate (TM > 0.5): {aggregated_metrics['success_rate_tm_gt_0.5']:.2%}")
        print(f"Success rate (TM > 0.7): {aggregated_metrics['success_rate_tm_gt_0.7']:.2%}")
        
        return report


class CompetitionEvaluator:
    """Evaluator specifically for competition format."""
    
    def __init__(self, competition_format: bool = True):
        self.competition_format = competition_format
    
    def evaluate_competition_submission(self,
                                     submission_coords: np.ndarray,
                                     true_coords: np.ndarray,
                                     sequence_lengths: List[int]) -> Dict[str, float]:
        """Evaluate competition submission format."""
        # Split submission coordinates by sequence
        pred_sequences = []
        true_sequences = []
        
        start_idx = 0
        for seq_len in sequence_lengths:
            # Extract 5 decoys for this sequence
            pred_decoys = []
            for decoy_idx in range(5):
                decoy_start = start_idx + decoy_idx * seq_len
                decoy_end = decoy_start + seq_len
                pred_decoys.append(submission_coords[decoy_start:decoy_end])
            
            pred_sequences.append(pred_decoys)
            
            # True coordinates (single structure)
            true_start = start_idx
            true_end = true_start + seq_len
            true_sequences.append(true_coords[true_start:true_end])
            
            start_idx += 5 * seq_len
        
        # Evaluate
        evaluator = StructureEvaluator()
        results = evaluator.evaluate_dataset(pred_sequences, 
                                           [RNAStructure("", coords, [], [], "") for coords in true_sequences])
        
        return results
    
    def create_leaderboard_entry(self,
                               team_name: str,
                               submission_coords: np.ndarray,
                               true_coords: np.ndarray,
                               sequence_lengths: List[int]) -> Dict:
        """Create leaderboard entry."""
        metrics = self.evaluate_competition_submission(submission_coords, true_coords, sequence_lengths)
        
        entry = {
            "team_name": team_name,
            "rank": None,  # To be filled by competition organizer
            "score": metrics["mean_best_of_5_tm"],  # Competition metric
            "tm_score": metrics["mean_tm_score"],
            "rmsd": metrics["mean_rmsd"],
            "gdt_ts": metrics["mean_gdt_ts"],
            "gdt_ha": metrics["mean_gdt_ha"],
            "lddt": metrics["mean_lddt"],
            "clashscore": metrics["mean_clashscore"],
            "n_sequences": len(sequence_lengths),
            "submission_timestamp": None  # To be filled
        }
        
        return entry


def benchmark_model(model,
                  test_sequences: List[str],
                  test_structures: List[RNAStructure],
                  output_dir: str = "benchmark_results") -> Dict:
    """Benchmark model performance on test set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate predictions
    predictions = []
    confidences = []
    
    print(f"Benchmarking on {len(test_sequences)} sequences...")
    
    for i, (sequence, true_structure) in enumerate(zip(test_sequences, test_structures)):
        print(f"Processing sequence {i+1}/{len(test_sequences)}: {sequence[:20]}...")
        
        # Predict (this would use the actual model)
        # For now, create dummy predictions
        n_residues = len(sequence)
        decoys = []
        decoy_confidences = []
        
        for j in range(5):
            # Slightly perturbed versions of true structure
            noise = np.random.normal(0, 0.5, (n_residues, 3))
            decoy = true_structure.coordinates[:, 0, :] + noise
            decoys.append(decoy)
            decoy_confidences.append(0.7 + 0.1 * j)
        
        predictions.append(decoys)
        confidences.append(decoy_confidences)
    
    # Evaluate
    evaluator = StructureEvaluator()
    report = evaluator.create_evaluation_report(predictions, test_structures, 
                                              output_dir / "benchmark_report.json",
                                              confidences)
    
    return report
