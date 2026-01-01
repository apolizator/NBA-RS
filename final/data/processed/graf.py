import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- SABİT PARAMETRELER (Kanka senin verdiğin değerler) ---
TOTAL_PLAYERS = 572
TEAMS_PER_PLAYER = 29
TOTAL_REAL_TRANSFERS = 158
SEARCH_SPACE = TOTAL_PLAYERS * TEAMS_PER_PLAYER  # 16,588 toplam ihtimal

def draw_unified_analysis():
    # Dosya isimleri (2024/2025 kontrolü ile)
    recs_file = 'team_player_recs_summer_2024_fixed.csv'
    if not os.path.exists(recs_file):
        recs_file = 'team_player_recs_summer_2025_fixed.csv'
        
    success_file = 'final_basari_analizi.csv'
    
    if not os.path.exists(recs_file) or not os.path.exists(success_file):
        print("⚠️ Hata: CSV dosyaları bulunamadı. Lütfen dosyaların script ile aynı klasörde olduğundan emin ol.")
        return

    try:
        # 1. VERİLERİ OKU (Header dahil satır sayısından 1 çıkartıyoruz)
        df_recs = pd.read_csv(recs_file)
        df_success = pd.read_csv(success_file)
        
        total_recs = len(df_recs)  # Pandas header'ı saymaz, o yüzden direkt veri satırıdır
        hits = len(df_success)
        
        # 2. TAVSİYE SİSTEMİ METRİKLERİ
        # Recall: 158 transferin yüzde kaçını yakaladık?
        recall = (hits / TOTAL_REAL_TRANSFERS) * 100
        # Precision: 1500 önerinin yüzde kaçı başarılı?
        precision = (hits / total_recs) * 100
        # F1-Score: Precision ve Recall dengesi
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 3. LİFT ANALİZİ (Rastgele Şansa Karşı Performans)
        # Hiçbir mantık kurmadan rastgele seçim yapsaydık kaç tane tuttururduk?
        expected_random_hits = (total_recs / SEARCH_SPACE) * TOTAL_REAL_TRANSFERS
        lift = hits / expected_random_hits if expected_random_hits > 0 else 0

        # --- GÖRSELLEŞTİRME ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plt.style.use('ggplot') # Temiz bir görünüm için

        # SOL GRAFİK: Standart Metrikler
        m_labels = ['Precision (Keskinlik)', 'Recall (Duyarlılık)', 'F1-Score']
        m_values = [precision, recall, f1]
        colors1 = ['#3498db', '#e74c3c', '#2ecc71'] # Mavi, Kırmızı, Yeşil
        
        bars1 = ax1.bar(m_labels, m_values, color=colors1, width=0.5)
        ax1.set_ylim(0, max(m_values) + 15)
        ax1.set_ylabel('Yüzde (%)', fontweight='bold')
        ax1.set_title('Recommender System Performans Metrikleri', fontsize=14, fontweight='bold', pad=20)
        
        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f'%{h:.2f}', ha='center', fontweight='bold', fontsize=11)

        # SAĞ GRAFİK: Rastgele Şans vs. Senin Modelin
        c_labels = ['Rastgele Tahmin\n(Beklenen)', 'Senin Modelin\n(Gerçekleşen)']
        c_values = [expected_random_hits, hits]
        colors2 = ['#bdc3c7', '#e74c3c'] # Gri ve Kırmızı
        
        bars2 = ax2.bar(c_labels, c_values, color=colors2, width=0.5)
        ax2.set_ylim(0, hits + 10)
        ax2.set_ylabel('Başarılı Hit Sayısı', fontweight='bold')
        ax2.set_title(f'Lift Analizi: Rastgele Şansa Karşı Başarı\n(Model {lift:.2f} Kat Daha Etkili)', 
                      fontsize=14, fontweight='bold', pad=20)
        
        for bar in bars2:
            h = bar.get_height()
            label_val = f'{h:.1f}' if bar.get_x() < 0.5 else f'{int(h)}'
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, label_val, ha='center', fontweight='bold', fontsize=11)

        # Bilgi Kutusu
        summary_txt = (f"Model Özeti:\n"
                       f"-------------------\n"
                       f"Toplam Oyuncu: {TOTAL_PLAYERS}\n"
                       f"Olasılık Havuzu: {SEARCH_SPACE:,}\n"
                       f"Gerçek Transfer: {TOTAL_REAL_TRANSFERS}\n"
                       f"Yapılan Öneri: {total_recs}\n"
                       f"Nokta Atışı: {hits}")
        
        plt.gcf().text(0.86, 0.5, summary_txt, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig('nba_recommender_unified_analysis.png', dpi=300)
        print("✅ Grafik başarıyla oluşturuldu: nba_recommender_unified_analysis.png")

    except Exception as e:
        print(f"❌ Beklenmedik hata: {e}")

if __name__ == "__main__":
    draw_unified_analysis()