# 🚀 Kaggle Training Guide for RNA 3D Folding Pipeline

## 📋 Overview

This guide provides step-by-step instructions for training the RNA 3D Folding Pipeline on Kaggle's platform using the Stanford RNA 3D Folding competition dataset.

## 🗂️ Dataset Structure

The competition dataset is located at:
```
/kaggle/input/competitions/stanford-rna-3d-folding-2/
```

### Available Files:
- `MSA/` - Multiple Sequence Alignment data
- `PDB_RNA/` - Protein Data Bank RNA structures
- `extra/` - Additional supplementary data
- `sample_submission.csv` - Sample submission format
- `test_sequences.csv` - Test sequences for inference
- `train_labels.csv` - Training labels (3D coordinates)
- `train_sequences.csv` - Training sequences
- `validation_labels.csv` - Validation labels
- `validation_sequences.csv` - Validation sequences

## 🛠️ Setup Instructions

### 1. Environment Setup

```bash
# Navigate to working directory
cd /kaggle/working/

# Clone/copy your DSLSF repository
# (If uploading as dataset, extract it here)
# Otherwise, git clone if available

# Navigate to project directory
cd DSLSF

# Install dependencies
pip install -r requirements.txt

# Install additional Kaggle-specific dependencies
pip install biopython
pip install tqdm
pip install tensorboard

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import biopython; print('BioPython installed')"
python -c "import rna_model; print('RNA Model loaded successfully')"
```

### 2. Directory Structure

Your working directory should look like:
```
/kaggle/working/DSLSF/
├── rna_model/
├── scripts/
├── hpc_training.py
├── requirements.txt
├── README.md
└── kaggle_training.md
```

## 🎯 Training Commands

### Basic Training

```bash
!python hpc_training.py \
  --config configs/training_config.json \
  --data_dir /kaggle/input/competitions/stanford-rna-3d-folding-2 \
  --output_dir /kaggle/working/outputs \
  --epochs 20 \
  --batch_size 32 \
  --gpu_ids 0 \
  --seed 42
```

### Advanced Training with Validation

```bash
!python hpc_training.py \
  --config configs/training_config.json \
  --pipeline_config configs/pipeline_config.json \
  --data_dir /kaggle/input/competitions/stanford-rna-3d-folding-2 \
  --output_dir /kaggle/working/outputs \
  --epochs 50 \
  --batch_size 16 \
  --gpu_ids 0 \
  --learning_rate 1e-4 \
  --validation_split 0.2 \
  --checkpoint_interval 5 \
  --seed 42
```

### Pre-training Only

```bash
!python scripts/pretraining.py \
  --config configs/pretraining_config.json \
  --data_dir /kaggle/input/competitions/stanford-rna-3d-folding-2 \
  --output_dir /kaggle/working/pretraining_outputs \
  --epochs 30 \
  --batch_size 32 \
  --seed 42
```

### Fine-tuning Only

```bash
!python scripts/finetuning.py \
  --config configs/finetuning_config.json \
  --data_dir /kaggle/input/competitions/stanford-rna-3d-folding-2 \
  --output_dir /kaggle/working/finetuning_outputs \
  --epochs 20 \
  --batch_size 16 \
  --seed 42
```

## 📁 Configuration Files

Create necessary configuration files in `configs/` directory:

### Training Config (`configs/training_config.json`)
```json
{
  "model": {
    "hidden_size": 512,
    "num_layers": 12,
    "num_heads": 8,
    "dropout": 0.1
  },
  "training": {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_steps": 1000,
    "max_steps": 100000,
    "gradient_clip_norm": 1.0
  },
  "data": {
    "max_sequence_length": 200,
    "train_split": 0.8,
    "val_split": 0.2
  }
}
```

### Pipeline Config (`configs/pipeline_config.json`)
```json
{
  "model_type": "transformer",
  "contact_threshold": 8.0,
  "geometry_loss_weight": 1.0,
  "contact_loss_weight": 0.5,
  "fape_loss_weight": 0.3
}
```

## 🎯 Training Pipeline

### Phase 1: Data Preprocessing
```bash
!python scripts/data_collection.py \
  --input_dir /kaggle/input/competitions/stanford-rna-3d-folding-2 \
  --output_dir /kaggle/working/processed_data \
  --process_msa \
  --process_structures
```

### Phase 2: Pre-training
```bash
!python scripts/pretraining.py \
  --config configs/pretraining_config.json \
  --data_dir /kaggle/working/processed_data \
  --output_dir /kaggle/working/pretrained_models \
  --epochs 30 \
  --batch_size 32
```

### Phase 3: Fine-tuning
```bash
!python scripts/finetuning.py \
  --config configs/finetuning_config.json \
  --data_dir /kaggle/working/processed_data \
  --output_dir /kaggle/working/fine_tuned_models \
  --pretrained_model /kaggle/working/pretrained_models/best_model.pt \
  --epochs 20 \
  --batch_size 16
```

### Phase 4: Validation
```bash
!python scripts/validation_experiments.py \
  --model_path /kaggle/working/fine_tuned_models/best_model.pt \
  --data_dir /kaggle/input/competitions/stanford-rna-3d-folding-2 \
  --output_dir /kaggle/working/validation_results
```

## 📊 Monitoring Training

### Check Training Progress
```bash
# Monitor training logs
tail -f /kaggle/working/logs/training.log

# Check GPU usage
nvidia-smi

# Check memory usage
free -h
```

### TensorBoard (if available)
```bash
# Start TensorBoard
tensorboard --logdir /kaggle/working/outputs/tensorboard --port 6006
```

## 🎯 Inference and Submission

### Generate Predictions
```bash
!python competition_submission.py \
  --model_path /kaggle/working/fine_tuned_models/best_model.pt \
  --test_file /kaggle/input/competitions/stanford-rna-3d-folding-2/test_sequences.csv \
  --output_dir /kaggle/working/submissions \
  --cache_dir /kaggle/working/cache
```

### Create Submission File
```bash
!python scripts/submission_formatting.py \
  --predictions /kaggle/working/submissions/predictions.json \
  --output_file /kaggle/working/submission.csv
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 8
   
   # Enable gradient checkpointing
   --gradient_checkpointing
   ```

2. **Dataset Not Found**
   ```bash
   # Verify dataset path
   ls /kaggle/input/competitions/stanford-rna-3d-folding-2/
   ```

3. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install torch torchvision torchaudio
   pip install biopython
   pip install scipy
   pip install tqdm
   pip install tensorboard
   ```

4. **Dataclass Mutable Default Error**
   ```bash
   # This has been fixed in the codebase
   # All dataclass fields now use field(default_factory=...)
   ```

### Performance Optimization

1. **Use Mixed Precision**
   ```bash
   --mixed_precision
   ```

2. **Enable Data Loading Optimization**
   ```bash
   --num_workers 4
   --pin_memory
   ```

3. **Gradient Accumulation**
   ```bash
   --gradient_accumulation_steps 4
   ```

## 📈 Expected Results

### Training Metrics
- **Loss**: Should decrease steadily over epochs
- **TM-Score**: Target > 0.7 on validation set
- **RMSD**: Should decrease as training progresses

### Resource Usage
- **GPU Memory**: ~6-8GB for batch_size=16
- **Training Time**: ~2-4 hours for 20 epochs
- **Disk Space**: ~5GB for models and outputs

## 🎯 Best Practices

1. **Start Small**: Begin with smaller batch sizes and fewer epochs
2. **Monitor Validation**: Check validation performance regularly
3. **Save Checkpoints**: Enable checkpoint saving to resume training
4. **Use Seeds**: Set random seeds for reproducibility
5. **Log Everything**: Keep detailed training logs for debugging

## 📝 Complete Training Script

```bash
#!/bin/bash

# Complete Kaggle Training Pipeline

echo "🚀 Starting RNA 3D Folding Pipeline Training on Kaggle"

# Setup
cd /kaggle/working/DSLSF
mkdir -p outputs logs configs

# Data preprocessing
echo "📁 Processing data..."
python scripts/data_collection.py \
  --input_dir /kaggle/input/competitions/stanford-rna-3d-folding-2 \
  --output_dir /kaggle/working/processed_data

# Pre-training
echo "🎯 Pre-training model..."
python scripts/pretraining.py \
  --config configs/pretraining_config.json \
  --data_dir /kaggle/working/processed_data \
  --output_dir /kaggle/working/pretrained_models \
  --epochs 30 \
  --batch_size 32 \
  --seed 42

# Fine-tuning
echo "🔧 Fine-tuning model..."
python scripts/finetuning.py \
  --config configs/finetuning_config.json \
  --data_dir /kaggle/working/processed_data \
  --output_dir /kaggle/working/fine_tuned_models \
  --pretrained_model /kaggle/working/pretrained_models/best_model.pt \
  --epochs 20 \
  --batch_size 16 \
  --seed 42

# Validation
echo "📊 Validating model..."
python scripts/validation_experiments.py \
  --model_path /kaggle/working/fine_tuned_models/best_model.pt \
  --data_dir /kaggle/input/competitions/stanford-rna-3d-folding-2 \
  --output_dir /kaggle/working/validation_results

# Generate submission
echo "📤 Generating submission..."
python competition_submission.py \
  --model_path /kaggle/working/fine_tuned_models/best_model.pt \
  --test_file /kaggle/input/competitions/stanford-rna-3d-folding-2/test_sequences.csv \
  --output_dir /kaggle/working/submissions

echo "✅ Training pipeline completed!"
echo "📊 Results saved in /kaggle/working/"
```

## 🎯 Success Metrics

Monitor these metrics during training:
- **Training Loss**: Should decrease consistently
- **Validation TM-Score**: Target > 0.7
- **Inference Time**: < 144 seconds per sequence
- **Memory Usage**: < 8GB GPU memory
- **Submission Score**: Aim for top 25% on leaderboard

Good luck with your RNA 3D folding training on Kaggle! 🚀
