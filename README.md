# Neural Networks are Decision Trees

This project is a reimplementation of the research paper [Neural Networks are Decision Trees](https://arxiv.org/abs/2210.05189) by Çağlar Aytekin. The paper demonstrates that any neural network with activation functions can be approximated by a decision tree. This project explores this equivalence through both **regression** and **classification** tasks, with a focus on visualization and analysis.

---

## Project Features
- Compare neural networks and decision trees for regression and classification tasks on synthetic datasets.
- Visualizations:
  - Neural network predictions vs. decision tree predictions.
  - Residual plots for both models.
  - Decision tree structure for regression.
  - Decision boundaries for the neural network and decision tree.
  - Classification tree structure.

### Hyperparameter Tuning
- Experiment with tree depth and other parameters to optimize the decision tree’s performance and mimic the neural network.

## How to Run

### Prerequisites
1. Install Python 3.8+.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Regression Task
1. Navigate to the `main/` folder:
   ```bash
   cd main
   ```
2. Run the main experiments:
   ```bash
   python main.py
   ```

---

## Research Paper

This project is based on the concepts described in:
- **[Neural Networks are Decision Trees](https://arxiv.org/abs/2210.05189)** by Çağlar Aytekin.

The paper demonstrates the theoretical equivalence of neural networks and decision trees, providing a novel perspective on machine learning models.

---

## Future Improvements

- Implement more datasets to generalize findings.
- Experiment with advanced tree models (e.g., Random Forests, Gradient Boosted Trees).
- Extend experiments to higher-dimensional classification problems.
- Experiment with various tree and neural network methods and parameters.

---

## Acknowledgments

Special thanks to the authors of the research paper for their contributions to the field of machine learning.
