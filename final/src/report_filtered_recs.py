#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_filtered_recs.py

Amaç:
- Normalde yüksek Fit Score alan ancak "UNTOUCHABLE_PLAYERS" listesinde olduğu için
  ana öneri dosyasında yer almayan oyuncuları raporlamak.
- Bu sayede "Eğer Jokic takaslanabilir olsaydı, en çok hangi takıma uyardı?"
  sorusuna cevap verebiliriz.

Çıktı:
- data/processed/filtered_recommendations.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from config import PROCESSED_DIR
from constants import UNTOUCHABLE_PLAYERS, TEAM_NAME_TO_ABBR

# Girdi ve Çıktı Yolları
TEAM_NEEDS_PATH = PROCESSED_DIR / "team_needs.csv"
PLAYER_PROFILES_WEIGHTED_PATH = PROCESSED_DIR / "player_profiles_weighted.csv"
OUT_PATH = PROCESSED_DIR / "filtered_recommendations.csv"

# Recommender ile aynı parametreler
TARGET_PLAYER_LAST_SEASON = 2024
MIN_GAMES = 30
MIN_MP = 500.0
REPORT_TOP_N = 20  # Sadece ilk 20'ye girebilecek potansiyeli olanları raporla

def load_data():
    if not TEAM_NEEDS_PATH.exists() or not PLAYER_PROFILES_WEIGHTED_PATH.exists():
        raise FileNotFoundError("Girdi CSV dosyaları bulunamadı!")
    return pd.read_csv(TEAM_NEEDS_PATH), pd.read_csv(PLAYER_PROFILES_WEIGHTED_PATH)

def clean_team_name(name: str) -> str:
    return name.replace("*", "").strip() if isinstance(name, str) else ""

def normalize_series_safe(s: pd.Series, min_target: float = 0.1, max_target: float = 1.0) -> pd.Series:
    s_min, s_max = s.min(), s.max()
    if s_max <= s_min:
        return pd.Series(min_target, index=s.index)
    norm = (s - s_min) / (s_max - s_min)
    return min_target + norm * (max_target - min_target)

def prepare_player_scores(df_players: pd.DataFrame) -> pd.DataFrame:
    # Recommender.py ile birebir aynı skorlama mantığı
    df = df_players.copy()
    df = df[(df["last_season"] == TARGET_PLAYER_LAST_SEASON) & (df["G"] >= MIN_GAMES) & (df["MP"] >= MIN_MP)]
    
    def col_s(n): return pd.to_numeric(df[n], errors="coerce").fillna(0.0) if n in df.columns else pd.Series(0.0, index=df.index)
    
    df["Defense_Score"] = (col_s("STL")*2) + (col_s("BLK")*1.5) + (col_s("DRB")*0.5)
    df["Spacing_Score"] = (col_s("3P")*1.5) + (col_s("3PA")*0.5)
    df["Efficiency_Score"] = (col_s("PTS")*1) + (col_s("AST")*1.5) - (col_s("TOV")*2)
    
    # Star Score
    rebs = col_s("TRB") if "TRB" in df.columns else col_s("DRB")
    df["Star_Score"] = (col_s("PTS") * 2.0) + (col_s("AST") * 1.0) + (rebs * 1.0)

    # PQ_raw
    df["PQ_raw"] = (df["Efficiency_Score"]*0.3) + (df["Defense_Score"]*0.2) + (df["Spacing_Score"]*0.2) + (df["Star_Score"]*0.3)

    for c in ["Defense_Score", "Spacing_Score", "Efficiency_Score", "PQ_raw"]:
       df[c+"_norm"] = normalize_series_safe(df[c], 0.1, 1.0)
    
    # MP Factor (Toplam dakika)
    df["MP_Factor"] = (col_s("MP") / 1000.0).clip(1.0, 2.0)
    return df

def main():
    print("🔍 Filtrelenen (Untouchable) oyuncular analiz ediliyor...")
    
    df_team_needs, df_p_raw = load_data()
    df_players_scored = prepare_player_scores(df_p_raw)
    
    filtered_rows = []

    for _, trow in df_team_needs.iterrows():
        t_abbr = trow["team_abbr"]
        if not t_abbr: continue
        
        n_sc = float(trow["Need_scoring"])
        n_def = float(trow["Need_defense"])
        
        # Bu takım için TÜM oyuncuların skorunu hesapla (Filtresiz)
        candidates = []
        for _, prow in df_players_scored.iterrows():
            # Kendi takımı hariç
            if prow["last_team"] == t_abbr: continue

            fit = (prow["Spacing_Score_norm"] * n_sc + 
                   prow["Defense_Score_norm"] * n_def + 
                   prow["PQ_raw_norm"] * 1.0) * prow["MP_Factor"]
            
            candidates.append({
                "player_name": prow["player_name"],
                "target_position": str(prow.get("primary_position", "UNK")).upper(),
                "fit_score": fit,
                "is_untouchable": prow["player_name"] in UNTOUCHABLE_PLAYERS
            })
            
        # Puan sırasına diz
        candidates.sort(key=lambda x: x["fit_score"], reverse=True)
        
        # Sıralamada nerede olduklarına bak
        for rank, cand in enumerate(candidates, start=1):
            # Sadece UNTOUCHABLE olanları ve TOP N'e girenleri raporla
            if cand["is_untouchable"] and rank <= REPORT_TOP_N:
                filtered_rows.append({
                    "team_abbr": t_abbr,
                    "player_name": cand["player_name"],
                    "target_position": cand["target_position"],
                    "potential_rank": rank,
                    "fit_score": round(cand["fit_score"], 4),
                    "filter_reason": "Untouchable"
                })

    if filtered_rows:
        df_out = pd.DataFrame(filtered_rows)
        df_out.to_csv(OUT_PATH, index=False)
        print(f"✅ Filtrelenen oyuncular raporu kaydedildi: {OUT_PATH}")
        print(f"📊 Toplam {len(df_out)} satır (Dokunulmaz olup Top-{REPORT_TOP_N} potansiyeline sahip eşleşmeler).")
    else:
        print("⚠️ Hiçbir untouchable oyuncu Top-20 potansiyeline giremedi veya liste boş.")

if __name__ == "__main__":
    main()