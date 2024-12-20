# SVM Optimization with GridSearchCV

This project demonstrates how to optimize Support Vector Machine (SVM) models using GridSearchCV while ensuring efficient runtime on larger datasets. The implementation focuses on the covertype dataset from OpenML and includes several strategies to improve performance and accuracy.

## Features

- Uses the `covertype` dataset with a reduced sample size for faster processing.
- Implements SVM with linear kernels for efficiency.
- Optimizes hyperparameters using GridSearchCV.
- Evaluates model accuracy across multiple randomized train-test splits.
- Outputs results as a CSV file and visualizes accuracy trends with a line plot.

## Setup Instructions

### Prerequisites

1. Python 3.7+
2. Required Python libraries:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - openml

Install dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib openml
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Run the script:
   ```bash
   python svm_optimization.py
   ```

3. Results will be saved as `SVM_results_optimized.csv` in the current directory.

### Using Google Colab

To leverage GPU acceleration:

1. Upload the script to Google Colab.
2. Enable GPU in the runtime settings:
   - Go to `Runtime > Change runtime type`.
   - Set `Hardware accelerator` to `GPU`.

3. Execute the code.

## Files

- `svm_optimization.ipynb`: Main script for running the SVM optimization.
- `SVM_results_optimized.csv`: CSV output containing the best accuracy and parameters for each train-test split.

## Visualization

The script generates a line plot showing accuracy trends across different train-test splits.

Example:

![Accuracy per Sample](accuracy_plot.png)

## Key Optimizations

1. **Subsampled Dataset**: Reduced dataset size to 10,000 samples for faster computation.
2. **Limited Hyperparameter Grid**: Focused on linear kernels and a smaller `C` range.
3. **Stratified K-Fold Cross-Validation**: Ensured balanced class distributions across folds for better performance.
4. **Simplified Cross-Validation**: Reduced cross-validation splits to 2 for efficiency.

## Results

The optimized approach significantly reduces runtime (5-10 minutes on Colab with GPU) while maintaining competitive accuracy. The results highlight the best hyperparameters for each split and the corresponding accuracy.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

## Author

Developed by [Your Name].
