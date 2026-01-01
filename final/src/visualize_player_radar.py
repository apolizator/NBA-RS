#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_player_radar.py

Amaç:
- Oyuncu profillerini interaktif olarak görüntülemek.
- Önce bir REFERANS TAKIM seçilir, grafiği çizilir.
- Ardından bir OYUNCU seçilir ve oyuncunun grafiği takımın grafiği üzerine bindirilir.
- Böylece oyuncunun takıma uygunluğu (veya takımın eksiklerini kapatıp kapatmadığı) görselleştirilir.

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
TEAM_STATS_PATH = PROCESSED_DIR / "team_stats_profiles.csv"
OUTPUT_DIR = PROCESSED_DIR / "visualizations"


def load_and_prep_data():
    if not INPUT_PATH.exists() or not TEAM_STATS_PATH.exists():
        raise FileNotFoundError("Gerekli veri dosyaları (oyuncu veya takım) bulunamadı.")
    
    print(f"📂 Veriler yükleniyor...")
    df = pd.read_csv(INPUT_PATH)
    df_teams = pd.read_csv(TEAM_STATS_PATH)

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
    
    # --- TAKIM VERİLERİNİ HAZIRLA (Aynı Metrikler) ---
    # Sadece son sezonu al (varsayılan 2025)
    if "season" in df_teams.columns:
        max_season = df_teams["season"].max()
        df_teams = df_teams[df_teams["season"] == max_season].copy()
    
    # Takım metrikleri (Oyuncu ile aynı mantıkta percentile rank)
    df_teams["Scoring_Rank"] = df_teams["PTS"].rank(pct=True) * 100
    df_teams["Rebounding_Rank"] = df_teams["TRB"].rank(pct=True) * 100
    df_teams["Playmaking_Rank"] = df_teams["AST"].rank(pct=True) * 100
    
    # Takım savunma skoru
    df_teams["Def_Raw"] = (df_teams["STL"] * 2.0) + (df_teams["BLK"] * 1.5) + (df_teams["DRB"] * 0.5)
    df_teams["Defense_Rank"] = df_teams["Def_Raw"].rank(pct=True) * 100
    
    df_teams["Shooting_Rank"] = df_teams["3P"].rank(pct=True) * 100
    df_teams["Efficiency_Rank"] = df_teams["FG%"].rank(pct=True) * 100
    
    return df, df_teams

def get_radar_values(row):
    """Verilen satırdan (oyuncu veya takım) radar değerlerini çeker."""
    return [
        row["Scoring_Rank"],
        row["Shooting_Rank"],
        row["Playmaking_Rank"],
        row["Efficiency_Rank"],
        row["Defense_Rank"],
        row["Rebounding_Rank"]
    ]

def draw_radar_comparison(ax, team_row, player_row=None):
    """Takım grafiğini çizer, eğer oyuncu varsa üzerine bindirir."""
    team_name = team_row["team"]
    values_team = get_radar_values(team_row)
    
    categories = ['Scoring', 'Shooting', 'Playmaking', 'Efficiency', 'Defense', 'Rebounding']
    
    # Radar matematiği
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] # 0'dan 2pi'ye açılar
    angles += angles[:1] # Kapatmak için başa dön
    
    values_team += values_team[:1]
    
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
    
    # --- 1. TAKIM ÇİZİMİ (MAVİ) ---
    ax.plot(angles, values_team, linewidth=2, linestyle='solid', color='#1976D2', label=team_name)
    ax.fill(angles, values_team, '#1976D2', alpha=0.2)
    
    title = f"REFERANS TAKIM: {team_name}"
    
    # --- 2. OYUNCU ÇİZİMİ (KIRMIZI - Varsa) ---
    if player_row is not None:
        player_name = player_row["player_name"]
        values_player = get_radar_values(player_row)
        values_player += values_player[:1]
        
        ax.plot(angles, values_player, linewidth=2, linestyle='solid', color='#E53935', label=player_name)
        ax.fill(angles, values_player, '#E53935', alpha=0.4)
        
        title += f"\nvs\n{player_name}"

    ax.set_title(title, size=14, weight='bold', y=1.08)
    
    # Legend (Açıklama) kutusu
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

def main():
    df_players, df_teams = load_and_prep_data()
    
    # İnteraktif mod
    plt.ion()
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, polar=True)

    print("\n" + "="*50)
    print("🏀 NBA TAKIM vs OYUNCU KARŞILAŞTIRMA SİSTEMİ")
    print("="*50)

    while True:
        # --- ADIM 1: REFERANS TAKIM SEÇİMİ ---
        print("\n--- ADIM 1: REFERANS TAKIM SEÇİMİ ---")
        teams_avail = sorted(df_teams["team"].unique())
        # Kolaylık olsun diye kısaltmaları da gösterelim (varsa) veya sadece isimleri
        # Burada team_stats_profiles.csv'de genelde tam isim yazar (Atlanta Hawks).
        
        print("Mevcut Takımlar:")
        # 5'erli yazdır
        for i in range(0, len(teams_avail), 5):
            print(" | ".join(teams_avail[i:i+5]))
            
        ref_team_input = input("\n👉 Referans Takım Adı (Tam veya parça, örn: 'Lakers'): ").strip()
        if ref_team_input.upper() in ["Q", "EXIT"]: break
        
        # Takım bul
        team_match = df_teams[df_teams["team"].str.contains(ref_team_input, case=False)]
        if team_match.empty:
            print("❌ Takım bulunamadı!")
            continue
        
        ref_team_row = team_match.iloc[0]
        print(f"✅ Seçilen Referans Takım: {ref_team_row['team']}")
        
        # Sadece takımı çiz
        draw_radar_comparison(ax, ref_team_row, player_row=None)
        plt.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)

        # --- ADIM 2: OYUNCU SEÇİMİ DÖNGÜSÜ ---
        while True:
            print(f"\n--- ADIM 2: OYUNCU SEÇİMİ (Referans: {ref_team_row['team']}) ---")
            print("(Ana menüye dönmek için 'back', çıkmak için 'q' yazın)")
            
            # Oyuncu hangi takımda?
            if "last_team" in df_players.columns:
                p_teams = sorted(df_players["last_team"].dropna().unique().astype(str))
                print(f"Oyuncu Takımları: {', '.join(p_teams)}")
            
            p_team_input = input("\n👉 Oyuncunun Takım Kısaltması (örn: GSW): ").strip().upper()
            if p_team_input in ["Q", "EXIT"]: return # Komple çık
            if p_team_input == "BACK": break # Takım seçimine dön
            
            team_players = df_players[df_players["last_team"] == p_team_input].sort_values("player_name")
            
            if team_players.empty:
                print("❌ Takım bulunamadı veya oyuncusu yok.")
                continue
                
            # Oyuncu Listesi
            players = team_players["player_name"].tolist()
            print(f"\n📋 {p_team_input} Kadrosu:")
            for i, p in enumerate(players, 1):
                print(f"{i}. {p}")
            
            p_input = input("\n👉 Oyuncu Numarası: ").strip()
            if p_input.upper() in ["Q", "EXIT"]:
                return
            
            if not p_input.isdigit():
                print("❌ Lütfen numara girin.")
                continue
                
            idx = int(p_input) - 1
            if 0 <= idx < len(players):
                selected_player = players[idx]
                print(f"🎨 Karşılaştırma çiziliyor: {ref_team_row['team']} vs {selected_player}...")
                player_row = df_players[df_players["player_name"] == selected_player].iloc[0]
                draw_radar_comparison(ax, ref_team_row, player_row)
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
