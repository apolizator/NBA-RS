import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# 1. Verileri yükle
df_all = pd.read_csv('team_player_recs_summer_2024_fixed.csv')
df_success = pd.read_csv('final_basari_analizi.csv')

# 2. Eşleştirme için anahtar sütun oluştur (Takım + Oyuncu)
df_all['key'] = df_all['team_abbr'] + "_" + df_all['player_name']
df_success['key'] = df_success['Team'] + "_" + df_success['Player']

# 3. Hedef değişkeni oluştur (Success listesinde varsa 1, yoksa 0)
df_all['target'] = df_all['key'].isin(df_success['key']).astype(int)

# 4. ROC Eğrisi Hesapla
# fit_score: Tahmin olasılığı yerine geçiyor
fpr, tpr, thresholds = roc_curve(df_all['target'], df_all['fit_score'])
roc_auc = auc(fpr, tpr)

# 5. Görselleştirme (Mühendis Stili)
plt.figure(figsize=(10, 7))
plt.style.use('dark_background') # Sunumuna uygun karanlık tema

plt.plot(fpr, tpr, color='#00ffcc', lw=3, label=f'NBA Model (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random Guess')

# Grafik Detayları
plt.title('NBA Trade Prediction - ROC Curve', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('False Positive Rate (Yanlış Alarm)', fontsize=12, color='#cccccc')
plt.ylabel('True Positive Rate (Doğru Yakalama)', fontsize=12, color='#cccccc')
plt.legend(loc="lower right", frameon=True, facecolor='black', edgecolor='white')
plt.grid(alpha=0.1)

# En iyi threshold noktalarından birini işaretle
plt.annotate('Optimal Point', xy=(0.2, 0.8), xytext=(0.4, 0.6),
             arrowprops=dict(facecolor='white', shrink=0.05))

plt.tight_layout()
plt.savefig('nba_final_roc.png', dpi=300)
print(f"✅ ROC eğrisi 'nba_final_roc.png' olarak kaydedildi.")
print(f"📈 Model Başarı Skoru (AUC): {roc_auc:.4f}")
plt.show()