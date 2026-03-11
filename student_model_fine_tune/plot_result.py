import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "lora_round2"
OUT_DIR = SCRIPT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

def load_training_data(data_dir: Path):
    log_history_path = data_dir / "trainer_log_history.json"
    summary_path = data_dir / "train_summary.json"
    
    with open(log_history_path, "r", encoding="utf-8") as f:
        log_history = json.load(f)
        
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    return pd.DataFrame(log_history), summary

df, summary = load_training_data(DATA_DIR)


def plot_loss_curve(df, summary, out_dir):
    loss_df = df[df["loss"].notna()].sort_values("step").copy()
    loss_df["loss_smooth"] = loss_df["loss"].rolling(window=5, min_periods=1).mean()

    plt.figure(figsize=(7.2, 4.6)) 
    plt.plot(loss_df["step"], loss_df["loss"], linewidth=1.2, label="Train loss", color="#1f77b4")
    plt.plot(loss_df["step"], loss_df["loss_smooth"], linewidth=2.0, label="Smoothed loss", color="#ff7f0e")
    
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"Round 1 LoRA Training | N={summary.get('num_samples', 'N/A')} | epochs={summary.get('num_train_epochs', 'N/A')}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(out_dir / "round2_exp1_train_loss_curve.png", dpi=300)
    plt.close()

plot_loss_curve(df, summary, OUT_DIR)


def plot_learning_rate_curve(df, out_dir):
    lr_df = df[df["learning_rate"].notna()].sort_values("step").copy()

    plt.figure(figsize=(7.2, 4.6)) 
    plt.plot(lr_df["step"], lr_df["learning_rate"], linewidth=1.4, markersize=3, color="#2ca02c")
    
    plt.xlabel("Training step")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(out_dir / "round2_exp1_learning_rate_curve.png", dpi=300)
    plt.close()

plot_learning_rate_curve(df, OUT_DIR)