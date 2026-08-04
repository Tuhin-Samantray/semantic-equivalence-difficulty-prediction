# Transformation-Aware Verification Difficulty Prediction for Semantic Code Equivalence

## Overview
This repository contains the implementation and experimental setup for predicting the computational difficulty of semantic equivalence verification before formal verification begins. 

By analyzing structural code features (AST, CFG, complexity metrics, and applied code transformations), our machine learning model estimates the runtime and resource overhead required by equivalence verifiers.

## Research Objectives
* Extract structural and transformation-level features from source code pairs.
* Build a dataset using execution statistics from semantic code verification.
* Train machine learning regression models to forecast verification difficulty.
* Evaluate predictions against standard runtime and solver metrics.

## Repository Structure
+-- data/                    # Raw, processed data, and verification logs
+-- docs/                    # Research notes and documentation
+-- experiments/             # Experimental scripts and Jupyter notebooks
+-- src/                     # Core Python source modules
¦   +-- ast_extractor/       # AST parsing utilities
¦   +-- cfg_builder/         # Control Flow Graph builders
¦   +-- feature_extraction/  # Feature engineering pipeline
¦   +-- models/              # ML training and evaluation models
+-- tests/                   # Unit test suite
+-- .gitignore               # Ignored files configuration
+-- README.md                # Project overview
