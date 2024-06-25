# Linear Discriminant Analysis (LDA) Data Mining

   This project demonstrates the implementation of LDA using various popular libraries such as [NumPy][https://numpy.org/], [SciPy][https://scipy.org/], [PyTorch][https://pytorch.org/], and [TensorFlow](https://www.tensorflow.org/).
   Each implementation is contained within its own [Jupyter Notebook](https://jupyter.org/), providing a comprehensive and detailed guide on how to perform LDA using these different tools.

## Repository Structure

   - [Implementation of LDA using NumPy](LDA_Implement_With_NumPy.ipynb)
   - [Implementation of LDA using SciPy](LDA_Implement_With_SciPy.ipynb)
   - [Implementation of LDA using PyTorch](LDA_Implement_With_PyTorch.ipynb)
   - [Implementation of LDA using TensorFlow](LDA_Implement_With_TensorFlow.ipynb)

## Dataset

   The dataset used in this project is the [`heart_statlog_cleveland_hungary_final.csv`](heart_statlog_cleveland_hungary_final.csv), which combines heart disease data from various sources.
   This dataset includes numerous attributes related to heart disease, and it is a common benchmark for evaluating classification algorithms.

## Requirements

   To run these notebooks, you will need the following libraries installed in your Python environment:

   | Library     | Version     | Implementation                  |
   |-------------|-------------|---------------------------------|
   | NumPy       | >= 1.21.0   | LDA_Implement_With_NumPy        |
   | Pandas      | >= 1.3.0    | All implementations             |
   | Matplotlib  | >= 3.4.2    | All implementations             |
   | PyTorch     | >= 1.9.0    | LDA_Implement_With_PyTorch      |
   | SciPy       | >= 1.7.0    | LDA_Implement_With_SciPy        |
   | TensorFlow  | >= 2.5.0    | LDA_Implement_With_TensorFlow   |

   You can install these dependencies using pip:

```bash
   pip install numpy scipy torch tensorflow pandas matplotlib
```

## Overview

   * Implement With NumPy

   This code details the step-by-step process of implementing LDA from scratch using NumPy.


      It covers the following steps:
      - Data preprocessing
      - Computing the mean vectors
      - Constructing the scatter matrices
      - Solving the eigenvalue problem for the scatter matrices
      - Selecting the linear discriminants
      - Transforming the dataset

   * Implement With SciPy

   This code shows how to leverage SciPy's linear algebra capabilities to implement LDA. 


      It includes:
      - Using SciPy for matrix operations
      - Simplifying eigenvalue decomposition with SciPy functions
      - Verifying the results against the NumPy implementation

   * Implement With PyTorch

   Here, we utilize PyTorch for implementing LDA, which is particularly useful for those familiar with deep learning frameworks. 


      This notebook covers:
      - Utilizing PyTorch tensors for data representation
      - Implementing LDA using PyTorch's linear algebra functions
      - Comparing performance and results with NumPy and SciPy implementations

   * Implement With TensorFlow

   This code demonstrates the implementation of LDA using TensorFlow. 


      It includes:
      - Using TensorFlow for data manipulation
      - Implementing the LDA algorithm with TensorFlow's high-level operations
      - Performance analysis and comparison with other implementations

## Results and Analysis

   Each code concludes with a section on results and analysis, where we evaluate the performance of the LDA implementations on the heart disease dataset.
   We visualize the transformed data and discuss the effectiveness of LDA in dimensionality reduction and classification.

## License

   This repository is licensed under the Apache License 2.0.
   See the [LICENSE](./LICENSE) file for more details.