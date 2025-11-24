# RR-FBTC: Rank-Revealing Functional Bayesian Tensor Completion



This repository provides the MATLAB implementation of the **Rank-Revealing Functional Bayesian Tensor Completion (RR-FBTC)** method, primarily designed for tensor completion with the capability of dynamic rank determination and incorporating functional/kernel-based prior knowledge.

## 🛠 Prerequisites



To execute the provided MATLAB code, the following external toolboxes are required:

1. **Tensor Toolbox for MATLAB:** Essential for basic tensor operations (e.g., `tenmat`, `kr`).
   - *Link:* https://www.tensortoolbox.org/
2. **Tensorlab:** While the core function avoids Tensorlab's complex functions, having it available might resolve potential dependencies on related utility functions in your execution environment.
   - *Link:* https://www.tensorlab.net/

Please download both toolboxes and add them to your MATLAB path before running `demo.m`.



## 📁 File Structure

The repository contains the following files:

```
.
├── BCTD.m            # The main function implementing the RR-FBTC algorithm.
├── demo.m            # A demonstration script showing how to use BCTD.m for tensor completion.
└── data.mat          # Example dataset (e.g., SSF data) used by the demo script.
```

## 🚀 Getting Started

Follow these steps to run the demonstration script:

1. **Download Dependencies:** Download and install the **Tensor Toolbox** and **Tensorlab**.

2. **Add to Path:** Start MATLAB and add the directories of the downloaded toolboxes to the MATLAB path (using the `addpath` command or the GUI).

3. **Execute Demo:** Run the `demo.m` script from the MATLAB command window:

   Matlab

   ```
   >> demo
   ```

The `demo.m` script will load the `data.mat` file, generate a partial observation (tensor completion setting), apply the RR-FBTC model, calculate the MAE and RMSE, and plot the original, observed, and reconstructed tensor slices.

## 📄 Reference

If you use this code in your research, please cite the following paper:

> Title: When Bayesian Tensor Completion Meets Multioutput Gaussian Processes: Functional Universality and Rank Learning.
>
> Journal: IEEE Transactions on Signal Processing (TSP).

