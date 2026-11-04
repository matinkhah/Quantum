# Variational Quantum Neural Networks (VQNN) & Interval-Valued VQNN (IVQNN) Benchmark Results

This repository contains the comprehensive experimental results, configuration files, predictions, formatted tables, and figures evaluating **Variational Quantum Neural Networks (VQNN)** and **Interval-Valued Variational Quantum Neural Networks (IVQNN)**. 

The evaluation spans benchmarks across key network intrusion detection datasets (**KDDCUP99** and **UNSW-NB15**), scaling properties, optimization trajectories (PSO vs. Adam), federated learning scenarios, forecasting, and architectural design choices.

---

## 📂 Repository Structure

```markdown
fivqnn_results/
├── config.json                     # Main experiment configurations and hyperparameter choices
├── design_choices.json             # Framework architectural configuration rules
├── environment.txt                 # Python environment dependencies and exact version definitions
├── logs.txt                        # Global execution logs for the ablation and benchmarking suites
├── paste_back.json                 # Metadata layout configurations for visual artifacts
│
├── tables/                         # Formatted tables in LaTeX (.tex) and CSV formats
│   ├── T_main_kddcup99.csv/.tex    # Primary classification performance on KDDCUP99
│   ├── T_main_unsw_nb15.csv/.tex   # Primary classification performance on UNSW-NB15
│   ├── T_ablations.csv/.tex        # Component-wise ablation results
│   ├── T_complexity.csv/.tex       # Circuit depth, parameter counts, and complexity metrics
│   ├── T_datasets.csv/.tex         # Benchmark datasets descriptive statistics
│   ├── T_hyperparameters.csv/.tex  # Hyperparameter exploration configurations
│   ├── T_xi.csv/.tex               # Impact analysis for parameter xi variations
│   ├── tab_efficiency.tex          # Training and inference runtime efficiency comparison
│   └── circuit_table_full.csv      # Complete circuit specification parameters
│
├── figures/                        # Evaluation plots and visual assets
│   ├── fig_fed_rounds.png/.pdf     # Federated learning convergence over rounds
│   ├── fig_roc_pr.png/.pdf         # ROC and Precision-Recall evaluation curves
│   ├── fig_barren_plateau.png/.pdf # Gradient variance and barren plateau analysis
│   ├── fig_reliability.png         # Model calibration and reliability curves
│   ├── fig_forecast.png/.pdf       # Forecasting trajectory fits and prediction intervals
│   └── fig_decision_landscape.png  # Loss landscapes and optimization surfaces
│
├── preds/                          # Model prediction arrays saved as zipped NumPy binaries (.npz)
│   ├── kddcup99__VQNN-Adam__*.npz  # Predictions for VQNN optimized using Adam on KDDCUP99
│   ├── kddcup99__IVQNN-PSO__*.npz  # Predictions for IVQNN optimized using Particle Swarm on KDDCUP99
│   ├── unsw_nb15__XGB-full__*.npz  # Baseline predictions using XGBoost on UNSW-NB15
│   └── unsw_nb15__MLP-full__*.npz  # Baseline predictions using Multilayer Perceptrons on UNSW-NB15
│
└── [CSV Metrics Data]              # Quantitative experiment runs and scaling metrics
    ├── E1_main.csv                 # Core benchmarking metrics across configurations
    ├── E1_significance.csv         # Statistical significance testing metrics
    ├── E2_federated.csv            # Distributed / Federated training metrics
    ├── E3_ablations.csv            # Component drop-out and framework ablation data
    ├── E4a_barren_plateau.csv      # Gradient vanish and barren plateau tracking
    ├── E5_complexity.csv           # Computational footprint versus layer depth
    ├── E6_gamma_delta.csv          # Hyperparameter sensitivity for interval variables
    └── E8_forecasting.csv          # Time-series prediction performance metrics
```

---

## 🚀 Key Experiments Evaluated

1. **E1: Main Performance Benchmarking**  
   Compares VQNN and IVQNN architectures directly against classical baselines (XGBoost, Random Forests, Multilayer Perceptrons, and Logistic Regression) on major tabular threat detection benchmarks. Includes statistical significance verification.
   
2. **E2: Federated Learning Optimization**  
   Evaluates distributed clients using decentralized data distributions (including Dirichlet partitioning with varying alpha settings) to study central-vs-federated performance gaps.

3. **E3: Architectural Ablation Studies**  
   Systematically isolates quantum embedding techniques, variational ansatz selection layers, and classical post-processing additions to measure performance contributions.

4. **E4: Barren Plateau & Convergence Mechanics**  
   Tracks gradient variance scaling properties relative to layer depth to test for barren plateau immunity, contrasting PSO (Particle Swarm Optimization) trajectories with Adam gradient descent.

5. **E5: Mathematical Complexity & Exponents**  
   Measures processing footprints, complexity ceilings, and structural layer ratios as parameter dimensions scale up.

6. **E8: Predictive Forecasting Trajectories**  
   Validates IVQNN's robustness under uncertainty bounds for temporal tracking and forecasting applications.

---

## 🛠️ Reproduction and Environment

To load the model predictions and evaluate the metrics, establish your workspace matching the defined ecosystem:

```bash
# Clone the repository
git clone https://github.com/your-username/fivqnn_results.git
cd fivqnn_results

# Install required python environment packages
pip install -r environment.txt
```

### Loading Predictions
Predictions are saved as `.npz` files containing ground-truth values (`y.npy`) and output probability distributions (`p.npy`). Load them inside python via:

```python
import numpy as np

# Load a specific experiment's prediction suite
data = np.load('preds/kddcup99__IVQNN-PSO[T=20]__f__s1.npz')
y_true = data['y']
y_pred = data['p']

print(f"Loaded prediction shapes: True Labels {y_true.shape}, Model Outputs {y_pred.shape}")
```

---

## 📊 Summary Figures

All generated visual evaluations, including ROC curves, optimization curves, federated trajectories, and loss decision landscapes, are available inside the `/figures` directory for easy inclusion in research manuscripts and presentation materials.