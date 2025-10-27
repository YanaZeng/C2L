# Temporal Confounded Causal Imitation Learning

A framework for Confounded Causal Imitation Learning (C2L). 
C2L is a two-stage imitation learning framework that combats confounding effects in expert demonstrations:
- Stage I: Identifies a valid instrumental variable (IV) from past observations.
- Stage II: Learns a policy with reduced bias using C2L and C2L*.

## Project Structure

```plaintext
.
├── src/                    # Source code directory
│   ├── experts/           # Expert policy training
│   │   └── train_experts.ipynb  # Expert training scripts for different environments
│   ├── learners/          # Policy optimization algorithms
│   │   ├── bc.py         # Behavioral Cloning implementation
│   │   ├── ccil.py       # CCIL base implementation
│   │   ├── ccil2.py      # CCIL variant implementation
│   │   ├── ccil_antbullet.py    # CCIL for Ant environment
│   │   └── ccil_halfcheetah.py  # CCIL for HalfCheetah environment
│   ├── indTest/          # IV identification implementations
│   ├── TCN_AIT_test.py   # TCN-based IV identification
│   ├── AIT_condition.py  # AIT test conditions
│   ├── models.py         # Neural network models
│   └── *_utils.py        # Environment-specific utilities
├── data/                   # Data storage directory
│   ├── ant_diff_distribution/  # Different distribution data for Ant
│   ├── ant_diff_hops/         # Different hops data for Ant
│   ├── ant_train_data/        # Training data for Ant
│   ├── hc_diff_distribution/  # Different distribution data for HalfCheetah
│   ├── hc_diff_hop/          # Different hops data for HalfCheetah
│   ├── hc_train_data/        # Training data for HalfCheetah
│   ├── ll_accuracy_distribution/ # Distribution accuracy for LunarLander
│   ├── ll_accuracy_hops/     # Hops accuracy for LunarLander
│   ├── ll_train_data/        # Training data for LunarLander
│   ├── output_accuracy/      # Accuracy output results
│   └── output_mse&j/         # MSE and J value output results
├── Figure/                 # Visualization results directory
│   ├── Accuracy_Figure/      # Accuracy related figures
│   └── MSE&J_Figure/         # MSE and J value related figures
├── AntBullet.py           # Ant environment training and evaluation
├── HalfCheetah.py         # HalfCheetah environment training and evaluation
├── Lunarlander.py         # LunarLander environment training and evaluation
├── Precision_Test_3_TCN.py      # 3-hop TCN accuracy testing
├── Precision_Test_diff_hops.py   # Different hops accuracy testing
├── Precision_Test_diff_distribution.py  # Different distributions accuracy testing
└── vis.ipynb              # Visualization notebook
```
## Generation of Training Data
The code to generate training date for different environments is included in `LunarLander.py` , `HalfCheetah.py`, and `AntBullet.py`.So, before run IV Indentification Tests, you need to generate the necessary training data first. Obtain different training data by controlling different parameters such as `hop` and `distribution` in `*_rollout` function. These data will be saved in `data/*_train_data/` directory.

## IV Identification Tests
To run IV identification tests under 3-hop TCN , navigate to the `src/indTest/` directory and execute the following command:
```bash
python Precision_Test_3_TCN.py
```

To run IV identification tests under different hops and distribution, execute the following command:
```bash
python Precision_Test_diff_hops.py
```
```bash
python Precision_Test_diff_distribution.py
```

These will generate the precision results for the TCN-based IV identification test.

## Policy Optimization
The framework implements three main imitation learning algorithms:
```plaintext
- Behavioral Cloning (BC)
- CCIL
- CCIL*
```
We train the three algorithms on the LunarLander, HalfCheetah, and AntBullet environments. To train these algorithms under environments, execute the following command:
```bash
python Lunarlander.py
```
```bash
python HalfCheetah.py
```
```bash
python AntBullet.py 
```

## Output Results
- All kinds of data,like training data, mse, J .etc, is saved in the corresponding environment folder under `data/` directory
- Visualization results are saved in `Figure/` directory:
  - `Accuracy_Figure/ `: Accuracy related figures
  - `MSE&J_Figure/` : MSE and J value related figures

## Environmental Support
The framework supports three main environments:
- LunarLander (Continuous control)
- HalfCheetah
- Ant (PyBullet)
Each environment has its specific utilities in corresponding `*_utils.py` files.

## Requirements
- Python 3.8.20
- PyTorch  2.4.1+cu124
- Stable-baselines3 2.4.0
- Gym 0.18.0
- PyBullet 3.2.5
- MuJoCo 1.50.1.0
- NumPy 1.24.4

## Acknowledgment
>This code builds upon the implementation of [Swamy et. al, 2022]. We thank the authors for their valuable work.
