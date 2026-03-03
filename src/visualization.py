import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(y_true, y_pred, title='Random Forest: Actual vs Predicted'):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Predictions')
    line_coords = [y_true.min(), y_true.max()]
    plt.plot(line_coords, line_coords, 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def residual_plot(y_test, test_preds, title = 'Residual Plot: CPU Temperature Prediction'):
    residuals = y_test - test_preds
    plt.figure(figsize=(10, 6))
    plt.scatter(test_preds, residuals, alpha=0.5, color='teal')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel('Predicted Temperature (°C)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def histogram(y_test, test_preds, title = 'Histogram of Prediction Errors'):
    errors = y_test - test_preds
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.title(title)
    plt.xlabel("Error Magnitude (Actual - Predicted)")
    plt.ylabel("Frequency (How many times this error occurred)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()