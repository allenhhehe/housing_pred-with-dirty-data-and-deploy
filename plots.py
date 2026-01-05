import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)

def plot_history(cost_history,out_path="reports/figures/cost_curve.png"):
    ensure_dir(os.path.dirname(out_path))

    plt.figure()
    plt.plot(cost_history)
    plt.title("Cost Curve (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig()
    plt.close

def plot_pred_scatter(y_true, y_pred, out_path="reports/figures/pred_scatter.png"):
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    plt.title("Predicted vs True")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)  
    plt.close()

def plot_residuals(y_true, y_pred, out_path="reports/figures/residuals.png"):
    ensure_dir(os.path.dirname(out_path))
    res = np.asarray(y_pred) - np.asarray(y_true)
    plt.figure()
    plt.hist(res, bins=50)
    plt.title("Residuals Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




