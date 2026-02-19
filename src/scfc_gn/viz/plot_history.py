import matplotlib.pyplot as plt
import numpy as np

from scfc_gn.train.history import TrainHistory


def plot_history(
        history: TrainHistory,
        baseline, pred_mean, pred_std,
        y_mean, y_std, corr
):
    plt.figure(figsize=(8, 5))

    epochs = np.array(history.epochs)
    train_losses = np.array(history.train_losses)
    val_losses = np.array(history.val_losses)


    # ---- Main curves ----
    plt.plot(epochs[1:], train_losses[1:], label="Train loss")
    plt.plot(epochs[1:], val_losses[1:], label="Validation loss")

    # ---- Baseline line ----
    baseline_line = [baseline] * len(epochs)
    plt.plot(epochs, baseline_line, linestyle="--", label="Zero baseline")

    # ---- Mark best validation epoch ----
    val_array = np.array(val_losses)
    best_idx = np.argmin(val_array)
    best_epoch = epochs[best_idx]
    best_val = val_array[best_idx]

    plt.scatter(best_epoch, best_val)

    plt.annotate(
        f"Best val: {best_val:.4f}\n@ epoch {best_epoch}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 5, best_val + 0.005),
        arrowprops=dict(arrowstyle="->")
    )

    # ---- Labels & title ----
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SC→FC Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)

    final_epoch = epochs[-1] if len(epochs) else None
    final_train = train_losses[-1] if len(train_losses) else float("nan")
    final_val = val_losses[-1] if len(val_losses) else float("nan")


    # ---- Pretty statistics box ----
    stats_text = (
        f"Final epoch {final_epoch}\n"
        f"{'-' * 25}\n"
        f"Train     : {final_train:.4f}\n"
        f"Val       : {final_val:.4f}\n"
        f"Baseline  : {baseline:.4f}\n"
        f"{'-' * 25}\n"
        f"Pred μ    : {pred_mean:.4f}\n"
        f"Pred σ    : {pred_std:.4f}\n"
        f"True μ    : {y_mean:.4f}\n"
        f"True σ    : {y_std:.4f}\n"
        f"Corr      : {corr:.3f}"
    )

    plt.gca().text(
        0.03, 0.97,  # top-left corner (relative coords)
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.9
        )
    )

    plt.tight_layout()
    plt.show()
