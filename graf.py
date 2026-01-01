import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- FIXED CONSTANTS ---
TOTAL_PLAYERS = 572
TEAMS_PER_PLAYER = 29
TOTAL_REAL_TRANSFERS = 158
SEARCH_SPACE = TOTAL_PLAYERS * TEAMS_PER_PLAYER  # 16,588 combinations
PROJECTS = ["random", "notime", "all1", "final"]

# --- KANKA BURAYI DEĞİŞTİR (RANDOM MANUEL DEĞERLERİN) ---
RANDOM_AVG_HITS = 9          # Ortalama isabet sayısı
RANDOM_AVG_RECS = 1050         # Toplam yapılan öneri sayısı (30 takım x 25 oyuncu gibi)
RANDOM_MANUAL_COVERAGE = 100  # <--- BURAYA İSTEDİĞİN YÜZDEYİ YAZ (Örn: 73.05 veya 100)
# -------------------------------------------------------

def calculate_advanced_metrics(proj_name):
    # --- RANDOM İÇİN MANUEL GİRİŞ BLOĞU ---
    if proj_name.lower() == "random":
        hits = RANDOM_AVG_HITS
        n_recs = RANDOM_AVG_RECS
        
        # Binary Classification Components
        tp = hits
        fp = n_recs - hits
        fn = TOTAL_REAL_TRANSFERS - hits
        tn = SEARCH_SPACE - (tp + fp + fn)

        # Metrics Calculations
        precision = (tp / n_recs) * 100 if n_recs > 0 else 0
        recall = (tp / TOTAL_REAL_TRANSFERS) * 100
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # MANUEL COVERAGE GİRİŞİ
        coverage = RANDOM_MANUAL_COVERAGE 
        
        accuracy = ((tp + tn) / SEARCH_SPACE) * 100
        specificity = (tn / (tn + fp)) * 100

        return {
            "name": "RANDOM (AVG)",
            "hits": hits,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "coverage": coverage,
            "accuracy": accuracy,
            "specificity": specificity
        }

    # --- DİĞER PROJELER İÇİN DOSYA OKUMA BLOĞU (DEĞİŞMEDİ) ---
    base_path = os.path.join(proj_name, "data", "processed")
    recs_file = os.path.join(base_path, "team_player_recs_summer_2024_fixed.csv")
    success_file = os.path.join(base_path, "final_basari_analizi.csv")
    
    if not os.path.exists(recs_file) or not os.path.exists(success_file):
        print(f"Warning: Missing files for {proj_name} at {base_path}")
        return None

    try:
        df_recs = pd.read_csv(recs_file)
        df_success = pd.read_csv(success_file)
        
        n_recs = len(df_recs)
        hits = len(df_success)
        unique_players_rec = df_recs['player_name'].nunique()

        tp = hits
        fp = n_recs - hits
        fn = TOTAL_REAL_TRANSFERS - hits
        tn = SEARCH_SPACE - (tp + fp + fn)

        precision = (tp / n_recs) * 100 if n_recs > 0 else 0
        recall = (tp / TOTAL_REAL_TRANSFERS) * 100
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        coverage = (unique_players_rec / TOTAL_PLAYERS) * 100
        accuracy = ((tp + tn) / SEARCH_SPACE) * 100
        specificity = (tn / (tn + fp)) * 100

        return {
            "name": proj_name.upper(),
            "hits": hits,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "coverage": coverage,
            "accuracy": accuracy,
            "specificity": specificity
        }
    except Exception as e:
        print(f"Error processing {proj_name}: {e}")
        return None

def draw_dashboard():
    results = [calculate_advanced_metrics(p) for p in PROJECTS]
    results = [r for r in results if r is not None]
    
    if not results:
        print("Error: No valid data found.")
        return

    metrics_to_plot = [
        ("Success Hits", "hits", "Blues", False),
        ("Precision (%)", "precision", "Greens", True),
        ("Recall (%)", "recall", "Reds", True),
        ("F1-Score (%)", "f1", "Purples", True),
        ("Coverage (%)", "coverage", "YlOrBr", True),
        ("Accuracy (%)", "accuracy", "GnBu", True),
        ("Specificity (%)", "specificity", "RdPu", True)
    ]

    names = [r['name'] for r in results]
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    axes = axes.flatten()

    for i, (title, key, cmap_name, is_pct) in enumerate(metrics_to_plot):
        ax = axes[i]
        values = [r[key] for r in results]
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.4, 0.8, len(names)))
        
        bars = ax.bar(names, values, color=colors, alpha=0.9, edgecolor='black', width=0.6)
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        for bar in bars:
            h = bar.get_height()
            label = f"{h:.2f}%" if is_pct else f"{int(h) if h == int(h) else h:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2, h + (max(values)*0.01), 
                    label, ha='center', fontweight='bold', fontsize=11)

    fig.delaxes(axes[7])
    plt.suptitle('NBA Multi-Model Evaluation Dashboard', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('nba_benchmarking_final.png', dpi=300, bbox_inches='tight')
    print(f"Success: Dashboard saved as 'nba_benchmarking_final.png'")

if __name__ == "__main__":
    draw_dashboard()