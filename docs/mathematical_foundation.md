# Mathematical Foundation of RNA 3D Folding Pipeline

This document describes the mathematical principles and formulations underlying the RNA 3D folding pipeline.

## Table of Contents

1. [Language Model Mathematics](#language-model-mathematics)
2. [Secondary Structure Prediction](#secondary-structure-prediction)
3. [SE(3)-Equivariant Geometry](#se3-equivariant-geometry)
4. [Attention Mechanisms](#attention-mechanisms)
5. [Loss Functions](#loss-functions)
6. [Coordinate Representations](#coordinate-representations)
7. [Optimization Methods](#optimization-methods)

## Language Model Mathematics

### Transformer Architecture

The RNA language model uses a transformer architecture with masked span objectives:

#### Input Embeddings

For a sequence of length $L$ with nucleotide tokens $x_i \in \{A, U, G, C, N\}$:

$$E = \text{Embedding}(X) \in \mathbb{R}^{L \times d}$$

$$E' = E + P$$

where $P$ is positional encoding:

$$P_{i,2j} = \sin\left(\frac{i}{10000^{2j/d}}\right)$$
$$P_{i,2j+1} = \cos\left(\frac{i}{10000^{2j/d}}\right)$$

#### Multi-Head Attention

For each attention head $h$:

$$Q_h = E'W_Q^h, \quad K_h = E'W_K^h, \quad V_h = E'W_V^h$$

$$\text{Attention}(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_h$$

$$\text{MultiHead}(E') = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O$$

#### Masked Span Language Modeling

For masked span prediction, we create span masks of length $l_{\text{span}}$:

$$P(l_{\text{span}}) = (1-p)^{l_{\text{span}}-1}p$$

The loss is computed only on masked positions:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\mathcal{M}\setminus i})$$

where $\mathcal{M}$ is the set of masked positions.

## Secondary Structure Prediction

### Contact Prediction

Contact prediction uses pairwise attention to predict base pairing:

$$C_{ij} = \text{softmax}\left(W_c \cdot [E_i; E_j; E_i \odot E_j]\right)$$

where $[;]$ denotes concatenation and $\odot$ element-wise multiplication.

#### Pseudoknot Prediction

For pseudoknot classification:

$$P_{\text{pk}}(i,j) = \text{softmax}\left(W_{\text{pk}} \cdot [E_i; E_j]\right)$$

### Top-K Hypothesis Generation

We generate $K$ secondary structure hypotheses using different sampling strategies:

$$\mathcal{H} = \{h_1, h_2, \ldots, h_K\}$$

Each hypothesis $h_k$ includes:
- Contact probabilities $C^{(k)}$
- Pseudoknot predictions $P_{\text{pk}}^{(k)}$
- Confidence scores $\alpha_k$

## SE(3)-Equivariant Geometry

### Rigid Body Transformations

#### Quaternion Representations

Rotation matrices are converted to quaternions for numerical stability:

$$q = \begin{bmatrix} q_w \\ q_x \\ q_y \\ q_z \end{bmatrix}, \quad \|q\| = 1$$

Conversion from rotation matrix $R$ to quaternion:

$$\text{trace}(R) = r_{11} + r_{22} + r_{33}$$

If $\text{trace}(R) > 0$:
$$q_w = \frac{1}{2}\sqrt{1 + \text{trace}(R)}$$
$$q_x = \frac{r_{32} - r_{23}}{4q_w}$$
$$q_y = \frac{r_{13} - r_{31}}{4q_w}$$
$$q_z = \frac{r_{21} - r_{12}}{4q_w}$$

#### Frame-Aligned Point Error (FAPE)

FAPE measures the alignment between predicted and true coordinate frames:

$$\mathcal{L}_{\text{FAPE}} = \frac{1}{N}\sum_{i=1}^{N} \left\| T_{\text{true},i}^{-1} T_{\text{pred},i} x_i - x_i^{\text{true}} \right\|_2$$

where $T_i$ is the transformation matrix for residue $i$.

### Invariant Point Attention (IPA)

IPA operates on 3D coordinates while maintaining equivariance:

$$\text{IPA}(X, F) = \sum_{j} \alpha_{ij} \cdot (F_j X_j + t_j)$$

where:
- $X_j$ are 3D coordinates
- $F_j$ are rotation matrices
- $t_j$ are translation vectors
- $\alpha_{ij}$ are attention weights

## Attention Mechanisms

### Sparse Attention

For long sequences, we use window-based attention to reduce complexity:

$$\mathcal{A}_{\text{window}}(i) = \{j : |i-j| \leq w/2\}$$

where $w$ is the window size (typically 64).

### Pairwise Attention for Secondary Structure

$$A_{ij} = \text{softmax}\left(\frac{(E_i W_Q)(E_j W_K)^T}{\sqrt{d}} + M_{ij}\right)$$

where $M_{ij}$ is a mask for invalid pairs.

## Loss Functions

### Multi-Task Loss

The total loss combines multiple objectives:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{LM}}\mathcal{L}_{\text{LM}} + \lambda_{\text{SS}}\mathcal{L}_{\text{SS}} + \lambda_{\text{geom}}\mathcal{L}_{\text{geom}} + \lambda_{\text{FAPE}}\mathcal{L}_{\text{FAPE}}$$

### Geometry Loss

#### Distance Loss

$$\mathcal{L}_{\text{dist}} = \sum_{i<j} \text{CE}(d_{ij}^{\text{pred}}, d_{ij}^{\text{true}})$$

where $d_{ij}$ are binned distances and CE is cross-entropy.

#### Angle Loss

$$\mathcal{L}_{\text{angle}} = \sum_{ijk} \text{MSE}(\theta_{ijk}^{\text{pred}}, \theta_{ijk}^{\text{true}})$$

#### Torsion Loss

$$\mathcal{L}_{\text{torsion}} = \sum_{ijkl} \text{MSE}(\phi_{ijkl}^{\text{pred}}, \phi_{ijkl}^{\text{true}})$$

### Confidence Loss

For TM-score prediction:

$$\mathcal{L}_{\text{conf}} = \text{MSE}(\hat{c}, \text{TM-score}(X^{\text{pred}}, X^{\text{true}}))$$

## Coordinate Representations

### Internal Coordinates

RNA structures are represented using internal coordinates:

- **Bond lengths**: $r_{i,i+1}$
- **Bond angles**: $\theta_{i,i+1,i+2}$
- **Dihedral angles**: $\phi_{i,i+1,i+2,i+3}$

### Cartesian to Internal Conversion

Given Cartesian coordinates $\{x_i\}$, internal coordinates are computed as:

$$r_{i,i+1} = \|x_{i+1} - x_i\|$$

$$\theta_{i,i+1,i+2} = \arccos\left(\frac{(x_i - x_{i+1}) \cdot (x_{i+2} - x_{i+1})}{\|x_i - x_{i+1}\| \|x_{i+2} - x_{i+1}\|}\right)$$

$$\phi_{i,i+1,i+2,i+3} = \arctan2\left(\frac{n_1 \cdot n_2}{n_1 \times n_2}\right)$$

where $n_1$ and $n_2$ are normal vectors of consecutive bond planes.

## Optimization Methods

### AdamW Optimizer

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

### Learning Rate Scheduling

Cosine annealing schedule:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t \pi}{T}\right)\right)$$

### Gradient Clipping

$$\text{if } \|g\|_2 > \text{clip\_norm}: g = g \cdot \frac{\text{clip\_norm}}{\|g\|_2}$$

## Numerical Stability

### Quaternion Normalization

To prevent drift in quaternion representations:

$$q \leftarrow \frac{q}{\|q\| + \epsilon}$$

where $\epsilon = 10^{-8}$ prevents division by zero.

### Matrix Conditioning

For rotation matrix orthogonalization:

$$R \leftarrow (I + \frac{1}{2}(R^T R - I))R$$

## Evaluation Metrics

### TM-Score

Template Modeling Score measures structural similarity:

$$\text{TM-score} = \frac{1}{L} \sum_{i=1}^{L} \frac{1}{1 + \left(\frac{d_i}{d_0(L)}\right)^2}$$

where $d_0(L) = 1.24(L - 15)^{1/3} - 1.8$.

### RMSD

Root Mean Square Deviation:

$$\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \|x_i^{\text{pred}} - x_i^{\text{true}}\|^2}$$

### GDT-TS

Global Distance Test Total Score:

$$\text{GDT-TS} = \frac{1}{4} \sum_{c \in \{1,2,4,8\}} \frac{N_c}{N}$$

where $N_c$ is the number of C-alpha pairs within distance threshold $c$ Å.

## References

1. Vaswani et al. (2017). "Attention Is All You Need"
2. Jumper et al. (2021). "Highly accurate protein structure prediction with AlphaFold"
3. Baek et al. (2021). "Accurate prediction of protein structures and interactions using a three-track neural network"
4. Townshend et al. (2022). "Diffusion models as a means of performing high-quality image synthesis"
