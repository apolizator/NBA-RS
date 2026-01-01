#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_player_radar.py

Amaç:
- Oyuncu profillerini interaktif olarak görüntülemek.
- Tkinter (GUI) kütüphanesi macOS üzerinde sürüm uyumsuzluğu hatası verdiği için,
  Terminal tabanlı interaktif seçim + Matplotlib penceresi yapısına geçildi.
- Sol panel yerine Terminal'den takım/oyuncu seçilir, grafik anlık güncellenir.

Kullanım:
  python src/visualize_player_radar.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi
from pathlib import Path

from config import PROCESSED_DIR

# Girdi ve Çıktı Yolları
INPUT_PATH = PROCESSED_DIR / "player_profiles_weighted.csv"
OUTPUT_DIR = PROCESSED_DIR / "visualizations"


def load_and_prep_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {INPUT_PATH}")
    
    print(f"📂 Veri yükleniyor: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    
    # Gerekli kolonların varlığını kontrol et (yoksa 0 doldur)
    # Beklenen: PTS, TRB, AST, STL, BLK, 3P, FG%, DRB (varsa)
    cols_to_numeric = ["PTS", "TRB", "AST", "STL", "BLK", "3P", "FG%", "DRB"]
    for c in cols_to_numeric:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # --- METRİKLERİ OLUŞTUR (0-100 Percentile Rank) ---
    # Ligdeki diğer oyunculara göre yüzdelik dilimi.
    # Örn: Scoring_Rank = 90 ise, oyuncu ligin %90'ından daha çok sayı atıyor demektir.

    # 1. Scoring (Skor Gücü): PTS
    df["Scoring_Rank"] = df["PTS"].rank(pct=True) * 100
    
    # 2. Rebounding (Ribaund): TRB
    df["Rebounding_Rank"] = df["TRB"].rank(pct=True) * 100
    
    # 3. Playmaking (Oyun Kurma): AST
    df["Playmaking_Rank"] = df["AST"].rank(pct=True) * 100
    
    # 4. Defense (Savunma): STL, BLK ve DRB ağırlıklı
    # Recommender mantığına benzer bir formül
    df["Def_Raw"] = (df["STL"] * 2.0) + (df["BLK"] * 1.5) + (df["DRB"] * 0.5)
    df["Defense_Rank"] = df["Def_Raw"].rank(pct=True) * 100
    
    # 5. Shooting (Şut Tehdidi): 3P İsabet Sayısı (Hacim)
    df["Shooting_Rank"] = df["3P"].rank(pct=True) * 100
    
    # 6. Efficiency (Verimlilik): FG% (Şut Yüzdesi)
    df["Efficiency_Rank"] = df["FG%"].rank(pct=True) * 100
    
    return df

def draw_radar(ax, df, player_name):
    # Veriyi çek
    row = df[df["player_name"] == player_name].iloc[0]
    
    categories = ['Scoring', 'Shooting', 'Playmaking', 'Efficiency', 'Defense', 'Rebounding']
    values = [
        row["Scoring_Rank"],
        row["Shooting_Rank"],
        row["Playmaking_Rank"],
        row["Efficiency_Rank"],
        row["Defense_Rank"],
        row["Rebounding_Rank"]
    ]
    
    # Radar matematiği
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    # Temizle ve Çiz
    ax.clear()
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, weight='bold')
    
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=8)
    ax.set_ylim(0, 100)
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#E53935')
    ax.fill(angles, values, '#E53935', alpha=0.4)
    
    ax.set_title(f"{player_name}\n(Percentile Rank)", size=14, weight='bold', y=1.08)

def main():
    df = load_and_prep_data()
    
    # İnteraktif mod
    plt.ion()
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, polar=True)
    plt.show()

    print("\n" + "="*50)
    print("🏀 NBA OYUNCU RADAR GRAFİĞİ (CLI MODU)")
    print("="*50)
    print("Çıkmak için 'q' veya 'exit' yazın.")

    while True:
        # 1. Takım Seçimi
        if "last_team" in df.columns:
            teams = sorted(df["last_team"].dropna().unique().astype(str))
            print(f"\nTakımlar: {', '.join(teams)}")
        else:
            print("Hata: 'last_team' kolonu yok.")
            break

        team_input = input("\n👉 Takım Kısaltması (örn: LAL): ").strip().upper()
        if team_input in ["Q", "EXIT"]:
            break
        
        team_players = df[df["last_team"] == team_input].sort_values("player_name")
        
        if team_players.empty:
            print("❌ Bu takım bulunamadı veya oyuncusu yok.")
            continue
            
        # 2. Oyuncu Seçimi
        players = team_players["player_name"].tolist()
        print(f"\n📋 {team_input} Kadrosu:")
        for i, p in enumerate(players, 1):
            print(f"{i}. {p}")
            
        p_input = input("\n👉 Oyuncu Numarası: ").strip()
        if p_input.upper() in ["Q", "EXIT"]:
            break
            
        if not p_input.isdigit():
            print("❌ Lütfen numara girin.")
            continue
            
        idx = int(p_input) - 1
        if 0 <= idx < len(players):
            selected_player = players[idx]
            print(f"🎨 Grafik çiziliyor: {selected_player}...")
            draw_radar(ax, df, selected_player)
            fig.canvas.draw()
            fig.canvas.flush_events()
            # Pencereyi öne getir (MacOS için gerekebilir)
            plt.pause(0.1)
        else:
            print("❌ Geçersiz numara.")

    print("👋 Çıkış yapıldı.")
    plt.close()

if __name__ == "__main__":
    main()
