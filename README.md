# KinPFN: Bayesian Approximation of RNA Folding Kinetics using Prior-Data Fitted Networks

Welcome to the official repository for **KinPFN**, a deep learning model designed to approximate the distributions of RNA first passage times.

## Overview

Understanding the kinetics of RNA folding is crucial for various applications in molecular biology. One important metric is the **First Passage Time**—the time it takes for an RNA molecule to reach a specific folded state for the first time. Computing the cumulative distribution function (CDF) of these times over a certain statistically significant number of simulations provides valuable insights into the folding process, including the likelihood of achieving a particular folded state within a given time frame. This helps researchers identify potential kinetic barriers and evaluate the overall efficiency of folding pathways. However, accurately simulating RNA folding kinetics is computationally expensive, requiring extensive computational resources and time. Therefore we introduce **KinPFN**, a deep learning model that approximates the CDF of RNA first passage times efficiently by using a very small set of real first passage times as context. For this work, we used the [Kinfold](https://www.tbi.univie.ac.at/RNA/Kinfold.1.html) algorithm as a reference for generating the real first passage times.

## Why KinPFN?

The most widely used method for simulating RNA folding kinetics is the [Kinfold](https://www.tbi.univie.ac.at/RNA/Kinfold.1.html) algorithm, which relies on the Gillespie algorithm, a powerful Monte Carlo simulation procedure. However, while effective, Kinfold is computationally intensive, making it less suitable for large-scale studies or real-time applications.

To address these challenges, we introduce **KinPFN**, a deep learning model that leverages [Prior-Data Fitted Networks (PFNs)](https://github.com/automl/PFNs) to approximate the CDFs of RNA first passage times efficiently. KinPFN is designed to provide a fast and accurate addition (not alternative) to common RNA kinetic simulators like Kinfold.

## Key Features

- **Deep Learning for RNA Folding Kinetics:** Trained on synthetic RNA folding times, KinPFN approximates the distributions of real first passage times accurately.
  
- **Efficient CDF Approximation:** Utilizes a minimal set of real folding times, processed in a single forward pass, to predict the posterior predictive distribution's CDF.

- **Pre-Trained Model:** KinPFN comes pre-trained, offering high accuracy in approximating real RNA first passage time distributions.


## Getting Started

### Installation

To use KinPFN, clone this repository and install the necessary dependencies. It's recommended to use a virtual environment, such as conda, for managing dependencies.  
The following commands assume that you have conda installed. If not, you can download it from the [official conda page](https://docs.conda.io/en/latest/miniconda.html).
```bash
git clone https://github.com/automl/KinPFN.git
cd KinPFN
conda create --name kinpfn_iclr_env python=3.9
conda activate kinpfn_iclr_env
pip install -r requirements.txt
pip install -e .
```
By running these commands, you'll set up a Python environment with all required dependencies and the KinPFN package. This environment allows you to execute the provided project notebooks seamlessly.

### Tutorial
See the [tutorial](./notebooks/kinpfn_tutorial.ipynb) for a guide on how to use KinPFN to approximate RNA first passage time distributions.


## Cite the work

KinPFN was published as a conference paper at the International Conference on Learning Representations (ICLR) 2025:
```
@inproceedings{
scheuer2025kinpfn,
title={Kin{PFN}: Bayesian Approximation of {RNA} Folding Kinetics using Prior-Data Fitted Networks},
author={Dominik Scheuer and Frederic Runge and J{\"o}rg K.H. Franke and Michael T. Wolfinger and Christoph Flamm and Frank Hutter},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=E1m5yGMOiV}
}
```

