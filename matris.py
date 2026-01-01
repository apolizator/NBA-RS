import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_confusion_matrix():
    # --- SENİN VERİLERİN (750 ÖNERİ MODELİ) ---
    # TP: 21, FP: 729
    # FN: 137, TN: 15701
    cm = np.array([[15701, 729], 
                   [137, 21]])

    # Etiketleri hazırlayalım
    group_names = ['True Negative (TN)', 'False Positive (FP)', 
                   'False Negative (FN)', 'True Positive (TP)']
    
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    
    # Yüzdesel oranlar (Toplam 16,588 üzerinden)
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    
    # Karelerin içine yazılacak metinleri birleştirelim
    labels = [f"{v1}\n\n{v2}\n\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    # Görselleştirme Ayarları
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2) # Yazı boyutu
    
    # Isı haritasını çizelim
    # 'Blues' renk paleti akademik sunumlar için en uygunudur
    ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                    xticklabels=['Predicted: NO', 'Predicted: YES'],
                    yticklabels=['Actual: NO', 'Actual: YES'])

    # Başlık ve eksen isimleri
    plt.title('Confusion Matrix: NBA Trade Recommender (N=750)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Algorithm Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Real-World Outcome', fontsize=14, fontweight='bold')
    
    # Resmi kaydet
    plt.tight_layout()
    plt.savefig('nba_confusion_matrix_750.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    draw_confusion_matrix()
    
    # Konsola metrik özetini de basalım
    tp, fp, fn, tn = 21, 729, 137, 15701
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    
    print("-" * 30)
    print(f"CONFUSION MATRIX SUMMARY:")
    print(f"Accuracy: %{accuracy:.2f}")
    print(f"Specificity: %{specificity:.2f}")
    print(f"Precision: %{precision:.2f}")
    print("-" * 30)