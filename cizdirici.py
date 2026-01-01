import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import glob
import os

# --- BURAYI GÜNCELLE ---
TARGET_COL = 'target'      # Gerçek sonuç kolonu (Örn: 'label', 'y', 'is_trade')
SCORE_COL = 'probability'   # Model olasılık kolonu (Örn: 'pred', 'score')
# -----------------------

def plot_nba_roc():
    folders = ['random', 'notime', 'all1', 'final']
    plt.figure(figsize=(10, 7))
    found_any = False

    for folder in folders:
        path = os.path.join(folder, 'data', 'processed', '*.csv')
        files = glob.glob(path)
        
        for file in files:
            df = pd.read_csv(file)
            
            # Kolon kontrolü ve Debug
            if TARGET_COL not in df.columns or SCORE_COL not in df.columns:
                print(f"\n❌ HATA: {file} içinde kolonlar bulunamadı!")
                print(f"🔍 Mevcut Kolonlar: {list(df.columns)}")
                continue
            
            fpr, tpr, _ = roc_curve(df[TARGET_COL], df[SCORE_COL])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f"{folder} (AUC = {roc_auc:.2f})")
            found_any = True

    if found_any:
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('NBA Trade Recommender ROC')
        plt.legend()
        plt.show()
    else:
        print("\n⚠️ Hiçbir dosya işlenemedi. Lütfen yukarıdaki 'Mevcut Kolonlar' listesine bakıp TARGET_COL ve SCORE_COL değişkenlerini güncelle.")

if __name__ == "__main__":
    plot_nba_roc()