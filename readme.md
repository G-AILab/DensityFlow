\# DensityFlow: Density-Guided Continuous Flow for Robust Counterfactual Explanations

This is a simplified PyTorch implementation of the paper "Density-Guided Continuous Flow for Robust Counterfactual Explanations" (DensityFlow).



DensityFlow addresses the challenge of \*\*model multiplicity\*\* in counterfactual explanations by:

\- Using \*\*Neural ODEs\*\* to generate continuous counterfactual paths

\- Employing \*\*Noise Contrastive Estimation (NCE)\*\* to guide paths through high-density regions

\- Producing counterfactuals that remain valid across multiple equally-performing models





\### Requirements

\- Python 3.8+

\- CUDA-compatible GPU (optional, but recommended)



\# Install dependencies



pip install -r requirements.txt



\# Quick Start



Execute the script to start the full training and evaluation pipeline:  python DensityFlow\_review\_version.py

