import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc

def plot_nba_master_roc_english():
    # 1. SCENARIOS AND FILE STRUCTURE
    folders = ['random', 'notime', 'all1', 'final']
    
    # Filenames are expected to be the same in each folder's directory
    UNIVERSE_FILENAME = 'team_player_recs_summer_2024_fixed.csv'
    SUCCESS_FILENAME = 'final_basari_analizi.csv'

    # Plot Styling (Cyberpunk / Dark Mode)
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8))
    # Tech Palette: Red, Orange, Cyan, Neon Green
    colors = ['#FF3131', '#FFBD00', '#00E5FF', '#39FF14'] 
    
    found_any = False

    print("--- Starting NBA Model Benchmark Analysis ---")

    # 2. ITERATE THROUGH EACH SCENARIO
    for folder, color in zip(folders, colors):
        # Construct paths for each scenario
        universe_path = os.path.join(folder, 'data', 'processed', UNIVERSE_FILENAME)
        success_path = os.path.join(folder, 'data', 'processed', SUCCESS_FILENAME)
        
        # Check if both files exist in the directory
        if not os.path.exists(universe_path) or not os.path.exists(success_path):
            print(f"Skipping: Files missing in directory [{folder}]")
            continue
            
        try:
            # Load Data
            df_uni = pd.read_csv(universe_path)
            df_suc = pd.read_csv(success_path)
            
            # Create Unique Keys for matching (Team + Player)
            df_uni['key'] = df_uni['team_abbr'].astype(str) + "_" + df_uni['player_name'].astype(str)
            df_suc['key'] = df_suc['Team'].astype(str) + "_" + df_suc['Player'].astype(str)
            
            # Labeling: 1 if hit is found in success list, 0 otherwise
            success_keys = set(df_suc['key'].unique())
            df_uni['target'] = df_uni['key'].apply(lambda x: 1 if x in success_keys else 0)
            
            # Calculate ROC and AUC
            # fit_score is used as the prediction probability
            fpr, tpr, _ = roc_curve(df_uni['target'], df_uni['fit_score'])
            roc_auc = auc(fpr, tpr)
            
            # Plot the curve for this specific model
            plt.plot(fpr, tpr, color=color, lw=3, label=f'{folder.upper()} Model (AUC = {roc_auc:.2f})')
            found_any = True
            
            print(f"Processed [{folder.upper()}]: Found {len(success_keys)} hits out of {len(df_uni)} recommendations.")

        except Exception as e:
            print(f"Error processing folder [{folder}]: {e}")

    if not found_any:
        print("Error: No data was processed. Check your file paths.")
        return

    # 3. FINAL PLOT DECORATION (ALL ENGLISH)
    plt.plot([0, 1], [0, 1], color='white', lw=1, linestyle='--', alpha=0.5) # Luck line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.title('NBA Trade Recommender - Multi-Model ROC Performance', fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    
    # Legend settings
    plt.legend(loc="lower right", frameon=True, facecolor='#111111', edgecolor='white', fontsize=10)
    plt.grid(color='#333333', linestyle=':', alpha=0.5)
    
    # Save output
    output_filename = 'nba_master_roc_english.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print("\n--- Analysis Complete ---")
    print(f"Graph saved as: {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_nba_master_roc_english()