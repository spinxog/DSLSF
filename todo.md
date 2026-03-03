# RNA 3D Folding Pipeline - Implementation TODO

## Phase 1: Foundation & Data Preparation
- [x] **Dataset Collection & Curation**
  - [x] Collect high-quality RNA structures from PDB
  - [x] Implement family-split validation (no train family in test)
  - [x] Create length-binned splits: short (<80nt), medium (80-200nt), long (>200nt)
  - [x] Build motif-enriched test sets (pseudoknots, junctions, ribozymes)
  - [x] Deduplicate at 80% identity using CD-HIT

- [x] **RNA Language Model Pretraining**
  - [x] Download RNAcentral database (~23.7M sequences)
  - [x] Implement masked span LM objective
  - [x] Add contact-prediction auxiliary head
  - [x] Train with family-balanced sampling (upweight rare/long RNAs)
  - [x] Save frozen embeddings for inference

## Phase 2: Core Model Architecture
- [x] **Secondary Structure Module**
  - [x] Implement SS predictor with top-k hypotheses (k=3)
  - [x] Add pseudoknot-aware prediction head
  - [x] Output soft probability matrices for each hypothesis

- [x] **Teacher Model (Large, Offline)**
  - [x] Build Evoformer-like pair encoder
  - [x] Implement SE(3)-equivariant structure module
  - [x] Add RNA-specific heads: sugar pucker, torsion angles, noncanonical interactions
  - [x] Multi-task outputs: frames, distances, angles, confidence scores
  - [x] Loss functions: FAPE, distogram cross-entropy, pucker classification

- [x] **Student Model (Compact, Inference)**
  - [x] Distill teacher to ≤150M parameters
  - [x] Implement sparse/axial attention for long sequences
  - [x] Lightweight SE(3)-lite blocks
  - [x] Ranking head for TM-score prediction

## Phase 3: Sampling & Diversity Generation
- [x] **Fast Sampler (Competition-Ready)**
  - [x] Implement MC-dropout stochastic passes
  - [x] SS hypothesis switching (top-3)
  - [x] MSA subsampling when available
  - [x] Generate ~20 decoys per sequence

- [x] **Advanced Sampler (Research)**
  - [x] Conditional diffusion/VAE for structural diversity
  - [x] Fragment assembly integration
  - [x] Latent space sampling for meaningful diversity

## Phase 4: Post-Processing & Ensemble
- [x] **Clustering & Selection**
  - [x] Implement decoy clustering by backbone RMSD
  - [x] Select 5 diverse representatives
  - [x] Use ranking head to order outputs

- [x] **Geometry Refinement**
  - [x] Internal-coordinate optimizer (1-3 iterations)
  - [x] Bond length/angle constraints
  - [x] Steric clash penalty

## Phase 5: Training & Validation
- [x] **Self-Distillation Pipeline**
  - [x] Ensemble agreement filtering (threshold TM > 0.7)
  - [x] Confidence-aware pseudo-labeling
  - [x] Physics-based vetting with fast energy proxy
  - [x] Label smoothing for ambiguous regions

- [x] **Validation Experiments**
  - [x] Family-split cross-validation
  - [x] Length-binned performance analysis
  - [x] Motif-specific benchmarks (pseudoknots, junctions)
  - [x] Ablation studies (LM vs no-LM, SS hypotheses, sampler types)
  - [x] Calibration analysis (predicted vs actual TM)

## Phase 6: Competition Deployment
- [x] **Notebook Optimization**
  - [x] Precompute and cache LM embeddings
  - [x] Bundle MSAs and fragment libraries
  - [x] Optimize for 8-hour runtime limit
  - [x] Implement mixed precision and quantization

- [x] **Submission Generation**
  - [x] Output 5 coordinate sets per sequence
  - [x] Format for competition requirements (C1' coordinates)
  - [x] Final ranking and selection

## Phase 7: Advanced Features (Stretch Goals)
- [x] **Template Integration**
  - [x] BLAST/Infernal homology search
  - [x] Template coordinate seeding
  - [x] Template-aware attention

- [x] **Multimodal Learning**
  - [x] SHAPE/DMS probing data integration
  - [x] Multi-task learning with experimental constraints

- [x] **Fragment Library**
  - [x] Motif-aware fragment mining
  - [x] Common junction and pseudoknot fragments
  - [x] Fragment assembly sampling

## Risk Mitigation Tasks
- [x] **Synthetic Data Safeguards**
  - [x] Curriculum weighting for pseudo-labels
  - [x] Disagreement filtering implementation
  - [x] Conservative pseudo-label acceptance

- [x] **Long RNA Handling**
  - [x] Sparse attention implementation
  - [x] Memory optimization for >200nt sequences
  - [x] Chunked processing pipeline

- [x] **Pseudoknot Specialization**
  - [x] Dedicated pseudoknot detection head
  - [x] Augmented training with known pseudoknots
  - [x] Noncanonical interaction modeling

## Phase 8: Advanced Optimizations & Competition Deployment

### A — Precompute & Bundle Artifacts (Storage, Embeddings, MSA Helpers)
- [x] **Cluster training/validation sequences into families**
  - [x] Cluster LM embeddings + length using HDBSCAN
  - [x] Keep cluster IDs and per-cluster stats
  - [x] Store cluster metadata for adaptive thresholds

- [x] **Compute family-adaptive contact-correlation thresholds**
  - [x] For each cluster, compute contact-correlation distribution from reconstruction tests
  - [x] Set T_corr_cluster = median - 2*MAD, clamp to [0.88, 0.98]
  - [x] Default T_corr = 0.90 for sequences with no cluster

- [x] **Build compressed LM embeddings for cached families**
  - [x] Compress per-residue embeddings via PCA → 128 dims
  - [x] Quantize to 8-bit
  - [x] Save reconstruction error metadata per sequence

- [x] **Compute sparse contact-residuals for flagged sequences**
  - [x] If contact-correlation < T_corr_cluster, compute top-K contact residuals
  - [x] K = min(128, round(0.05 * L^2_heuristic))
  - [x] Store (i,j,Δ) as 16-bit floats in bit-packed format
  - [x] Include verification hash

- [x] **Create motif library packaged for fast access**
  - [x] Group motifs by size and type
  - [x] Store in LMDB with contiguous blocks
  - [x] Include motif metadata (type, length, confidence, PDB-derived stats)

- [x] **Quantize motif library where safe**
  - [x] Keep high-value motif templates uncompressed
  - [x] Compress remaining with PCA+8-bit
  - [x] Provide index for retrieval

- [x] **Build ANN index for retrieval**
  - [x] Build IVF/OPQ index on compressed embeddings for entire corpus
  - [x] Store index with appropriate parameters (R=64 retrieval)

- [x] **Prepare distilled fallback LM adapter set**
  - [x] Include distilled LM ~75–100M params for in-notebook fallback
  - [x] Prepare tiny motif-awareness adapters for loading on demand

- [x] **Precompute MSA summaries for cached sequences**
  - [x] Store shallow MSA summaries: top covariation PCs
  - [x] Store top coevolving pairs and MSA depth
  - [x] Keep summaries compact

- [x] **Implement retrieval prefetch plan & LRU cache design**
  - [x] Batch-prefetch motif blocks for entire test batch
  - [x] Maintain LRU cache capacity for ~1000 fragments
  - [x] Use asynchronous prefetch

- [x] **Bundle size check & pruning policy**
  - [x] Ensure final bundle ≤ 15 GB
  - [x] Implement pruning: remove least-used families, reduce K to 64, compress more aggressively

### B — Input Processing (Per Test Sequence Initial Steps)
- [x] **Load cached artifact or fallback**
  - [x] If sequence in cache, load compressed embedding and sparse residuals
  - [x] Else: use distilled LM fallback to compute embeddings

- [x] **Fallback embedding retrieval augmentation**
  - [x] Retrieve top R=64 ANN neighbors
  - [x] Cluster into C=4 clusters; sample S=3 per cluster (12 total)
  - [x] Compute contact_agreement score vs query contact-head
  - [x] Reweight neighbors by w = 0.6*seq_sim + 0.4*contact_agreement
  - [x] If average contact_agreement < 0.3, trigger micro-online sampling (≤10s)

- [x] **Compute shallow MSA (only if not cached)**
  - [x] Run progressive MSA: quick search 10–30s
  - [x] If found <50 sequences and contact entropy high, run up to 60s total
  - [x] Limit depth to 200 sequences

- [x] **Produce hybrid features**
  - [x] Combine: compressed embedding (or fallback), sparse residual pairs
  - [x] Add MSA summaries (or pseudo-MSA augmentation)
  - [x] Add top-3 secondary-structure hypotheses with pair-prob matrices

- [x] **Compute complexity & entanglement scores**
  - [x] Complexity = function(length, number_of_multiway_junctions, predicted_contact_entropy)
  - [x] Entanglement = normalized crossing count of high-confidence pairs
  - [x] Save both for budgeting

### C — Contact, Domain, and Graph Preprocessing
- [x] **Predict pairwise contact probabilities (LM-head + MSA head + SS-based)**
  - [x] Produce ensemble of contact maps
  - [x] Combine into consensus with per-pair confidence

- [x] **Compute contact-graph signatures & hub candidates**
  - [x] Compute degree, betweenness centrality
  - [x] Select hubs: top M=4 by degree + top M2=2 by betweenness
  - [x] Add up to S=4 sentinel residues (low LM entropy or MI peaks)

- [x] **Detect entanglement / pseudoknots**
  - [x] If crossing density > 0.02 or planarity test fails, mark as entangled

- [x] **Produce N_domain_proposals (ensemble)**
  - [x] If entangled: generate up to 7–8 domain proposals focused on junction splits
  - [x] Else baseline 3–5 proposals
  - [x] Use spectral clustering + greedy community detection at multiple resolutions

- [x] **For each domain proposal compute domain complexity and expected time budget**
  - [x] Assign per-proposal priority using cluster-level complexity and inter-domain contact strength

### D — Student Model Inference & Coarse Folding (Domain-Level)
- [x] **Run student encoder per domain proposal**
  - [x] Student architecture: local windowed attention + M hub tokens
  - [x] SE(3)-local blocks on small neighborhoods only
  - [x] Output frames, continuous torsions + variance, distogram, per-residue confidence, predicted-TM proxy

- [x] **Use injected sparse residual pairs in pair features**
  - [x] Ensure attention/pair encodings include top-K contact residuals from precompute stage

- [x] **Apply hybrid attention design**
  - [x] Local attention window w=64
  - [x] Global hub attention with M_total≈8 tokens per domain
  - [x] Use long-range injection of high-confidence pairs

- [x] **Produce coarse domain decoys**
  - [x] Convert frames/torsions to coarse C1′ coordinates for domain-level decoys

- [x] **Domain-level verification**
  - [x] Compute per-domain contact-satisfaction (fraction of predicted high-confidence contacts satisfied)
  - [x] Compute torsion_penalty
  - [x] If domain-level contact satisfaction ≥ 0.8, mark domain decoy as high-confidence

- [x] **Budgeted early exit**
  - [x] If first domain proposal yields per-domain contact satisfaction ≥ 0.85 and low torsion_penalty, skip remaining proposals for that domain

### E — Topology-Aware Sampler (Local and Global)
- [x] **If entangled or complexity high, run targeted local topology exploration**
  - [x] Define search window W = min(120, 0.4 * region_length)
  - [x] Allow N_local_props up to 20

- [x] **Run graph-edit proposal generator**
  - [x] Operators: rewire pair, split/join junction connectivity, swap stems, insert/delete hairpin
  - [x] Generate proposals prioritized by predicted contact entropy reduction

- [x] **Parallel-tempering MCMC run**
  - [x] n_chains=4, temperatures = [1.0,1.6,2.6,4.2]
  - [x] max_steps_per_chain=500, swap_every=10
  - [x] Adaptive adjust temps to keep acceptance in [0.2,0.5]
  - [x] Total sampler time capped t_sampler_max = min(0.4*T_seq_budget, 240s)

- [x] **For each accepted proposal, perform mini constrained relaxation for grafts**
  - [x] Mini-MD: N_md=100 steps
  - [x] Then 5 normal-mode smoothing steps for local adjustments
  - [x] Reject if clash_score > C_max threshold

- [x] **For non-entangled/low complexity sequences run default sampler**
  - [x] Generate baseline 12 topology proposals via combination of top-3 SS hypotheses × MSA subsample × MC-dropout
  - [x] **Maintain acceptance-rate & diversity logging**
  - [x] Track acceptance rates per chain, swaps count, distinct topology clusters

### F — Stitched Domain Assembly & Cross-Domain Checks
- [x] **For domain-level decoys attempt docking**
  - [x] Dock per-domain decoys using coarse rigid-body + junction torsion adjustments
  - [x] For each assembly, compute inter-domain contact-satisfaction

- [x] **Cross-domain pseudoknot detection & merging**
  - [x] If crossing density across domains > 0.02, attempt to merge implicated domains
  - [x] Refold merged block, with size cap S_max=400 nt
  - [x] If implied merge > S_max, perform prioritized subgraph merging around top crossing regions

- [x] **Generate stitched decoys**
  - [x] For each sequence create up to k_stitch=8 stitched candidates
  - [x] Use top domain decoys combos
  - [x] Score them via rescoring consensus

- [x] **If L > 500 nt, switch to partial-output mode**
  - [x] Produce domain-level decoys and up to 2 stitched "best-effort" full-length candidates
  - [x] Tag outputs partial=true in metadata

### G — Relaxer, Rescoring, and Promotion Rules
- [x] **Run two-stage relaxation**
  - [x] Stage 1: coarse relax to enforce contacts and remove clashes (fast)
  - [x] Stage 2: local high-resolution relax only on regions with low torsion variance but detected clashes (bounded to small neighborhoods and time)

- [x] **Compute rescoring ensemble**
  - [x] Scores: knowledge-based statistical potential; rescoring net output; torsion-strain metric; contact-satisfaction; Rg deviation; inter-helix orientation metric
  - [x] Aggregate with calibrated weights learned offline
  - [x] Require at least 2 scoring systems to agree for promotion

- [x] **Rescoring net details**
  - [x] Small MLP/CNN trained offline with adversarial negatives + domain-adversarial loss
  - [x] Output predicted-TM mean + variance
  - [x] Per-decoy inference cost target ≤ 50 ms

- [x] **Two-stage promotion to top-5**
  - [x] Stage: all decoys pass soft acceptance if rescoring ensemble in top half
  - [x] Promotion: to top-5 require rescoring consensus above motif-type floors (lookup table), or satisfy topology-first fallback (see below)

### H — Clustering, Diversity Enforcement, Ranking & Calibration
- [x] **Cluster decoys by topology signature + RMSD**
  - [x] Topology signature = stem connectivity matrix compressed to fingerprint
  - [x] Cluster to identify unique topologies first, then cluster by RMSD within topologies

- [x] **Bayesian hierarchical calibration**
  - [x] Calibrate predicted-TM outputs using hierarchical model: motif-type → length-bin → domain-count
  - [x] Estimated via empirical Bayes
  - [x] Use posterior mean and variance for ranking

- [x] **Topology-first fallback rule**
  - [x] If a decoy is topologically unique (graph distance > threshold) AND contact-satisfaction > 0.55 AND predicted-TM variance high → allow at most 1 topology-first candidate into top-5 even if mean predicted-TM < motif floor

- [x] **Select top-5**
  - [x] Ensure selection includes: top-ranked decoy, up to 1 topology-first unique decoy (if condition met), and up to 3 diverse representatives from 2 or more topology clusters if available
  - [x] Enforce a maximum of 2 low-confidence decoys

- [x] **Mark decoys with metadata**
  - [x] For each selected decoy include: predicted-TM mean & variance, topology-cluster id, partial/full flag, time used, rescoring breakdown

### I — Submission Formatting, Sanity Checks, and Final Pass
- [x] **Per-decoy sanity checks**
  - [x] No NaNs; all residues have coordinates
  - [x] No pairwise C1′ distances < 1.5 Å
  - [x] Average bond-lengths within allowed ranges
  - [x] Predicted-TM proxy > 0.1 for at least one decoy

- [x] **If sanity check fails for a sequence**
  - [x] Attempt one quick resample (same budget ≤ 30s)
  - [x] If still fails, output domain-level decoys or a simple linearized chain as fallback and mark as failed in metadata

- [x] **Write submission.csv**
  - [x] For each residue include C1′ coords for decoys 1–5
  - [x] Include a separate metadata file explaining partial decoys
  - [x] Ensure formatting exactly matches competition spec

### J — Monitoring, Logging, Reproducibility & Post-Run Diagnostics
- [x] **Per-sequence scoreboard logging**
  - [x] Log: sequence id, L, complexity, entanglement_score, t_budget, t_used, number of domain proposals, sampler_stats, rescoring values, selected decoys summary

- [x] **Global runtime budget monitor**
  - [x] Continuously compute total elapsed vs allocated
  - [x] If projected exceed, trigger conservative mode (reduce sampler steps, reduce proposals) for remaining sequences

- [x] **Deterministic logging**
  - [x] Record RNG seeds for each decoy generation
  - [x] Save model versions and artifact hashes

- [x] **Post-run automated reports**
  - [x] Produce summary: distribution of predicted-TM, decoy topologies counts, fraction of partial outputs, sequences that triggered fallback LM, metrics per-length bin
  - [x] Highlight sequences with high variance for offline analysis

### K — Offline Training / Data Augmentation Tasks (for Teacher, Rescoring, Motif Models)
- [x] **Train rescoring network**
  - [x] Train on mixture of experimental decoys, teacher outputs, and adversarial negatives
  - [x] Include domain-adversarial loss
  - [x] Save small model for in-notebook inference

- [x] **Train motif-detector & adapters**
  - [x] Train compact motif detector (CNN/transformer adapter) to flag rare motif signatures
  - [x] Also train small motif-aware adapters for distilled LM

- [x] **Active motif augmentation**
  - [x] For motif classes with < N_min = 20 examples, run curated teacher+fragment assembly offline
  - [x] Create high-quality synthetic examples
  - [x] Validate via rescoring consensus before adding

- [x] **Build motif-type rescoring floors table**
  - [x] From validation, compute promotion floors per motif-type for rescoring
  - [x] Store lookup table

### L — Thresholds & Parameter Reference (Copyable)
- [ ] **Key thresholds / parameters**
  - [ ] T_corr_cluster clamp: [0.88,0.98]; default novel = 0.90
  - [ ] K sparse residuals: min(128, 0.05*L^2_heuristic)
  - [ ] ANN retrieval R=64, clusters C=4, samples S=3 per cluster
  - [ ] M hubs per domain: 4 (degree) + 2 (betweenness) + up to 4 sentinels → ~8 total
  - [ ] Entanglement crossing density flag > 0.02
  - [ ] Domain proposal counts: baseline 3–5; entangled up to 7–8; local exploration up to 20 region proposals
  - [ ] Parallel tempering temps: [1.0,1.6,2.6,4.2], acceptance target 20–40%
  - [ ] Mini-MD steps per graft: 100; normal-mode smoothing steps: 5
  - [ ] Max sampler time per sequence: t_sampler_max = min(0.4*T_seq_budget, 240s)
  - [ ] Stitch attempts k_stitch = 8; merge cap S_max = 400 nt; hard cutoff for partial-mode L > 500 nt
  - [ ] Rescoring promotion floor default TM ~0.25; motif floors stored in lookup table
  - [ ] Minimum per-seq time T_seq_min = 60s; max T_seq_max = 900s (per-budget rules)

### M — Final Acceptance & Operational Rules
- [ ] **Global early-abort rule**
  - [ ] If global runtime consumed > 95% of allocated before processing all sequences, abort remaining heavy operations and produce at least 2 high-confidence student-fast decoys per remaining sequence

- [ ] **Debugging & failure triage process**
  - [ ] For any sequence with predicted-TM variance > 0.25 or post-run low best-of-5 TM on validation, reconstruct seeds and reproduce offline for deeper teacher/generative exploration

## Implementation Priority (Immediate Actions)

### Priority 1 (Day 0-3) - High Impact, Low Cost
- [x] Sparse contact-residual storage + family T_corr
- [x] 75-100M distilled LM + retrieval reweighting
- [x] Progressive MSA builder with time caps
- [x] Per-sequence reconstruction QC metrics

### Priority 2 (Day 3-10) - Core Robustness
- [x] Entanglement detector + local BFS topology sampler
- [x] Ensemble domain proposals with adaptive pruning
- [x] Multi-hub + sentinel residue system
- [x] Parallel tempering MCMC implementation

### Priority 3 (Day 10-18) - Quality & Calibration
- [x] Consensus rescoring network + torsion-strain metrics
- [x] Mini-MD + normal-mode smoothing for grafts
- [x] Bayesian hierarchical calibration
- [x] Topology-first fallback rules

### Priority 4 (Day 18-28) - Advanced Features
- [x] Cross-domain pseudoknot detection + merging
- [x] Assembly-aware stitching algorithm
- [x] Global runtime allocation optimizer
- [x] Full monitoring and validation system

## Success Metrics
- [ ] **Primary**: Best-of-5 TM-score on validation
- [ ] **Secondary**: Median TM across length bins
- [ ] **Calibration**: Predicted vs actual TM correlation > 0.6
- [ ] **Robustness**: Performance on rare motifs
- [ ] **Efficiency**: < 144 seconds per sequence (for 200 sequences)
- [ ] **Bundle Size**: ≤15GB total compressed artifacts
- [ ] **Memory**: Peak GPU memory < 8GB during inference
- [ ] **Coverage**: >95% of sequences processed without fallback
