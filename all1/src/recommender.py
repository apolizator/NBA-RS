#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse
import sys
import random
import numpy as np
import pandas as pd

# Mevcut config yapına sadık kalarak yolları ayarlıyoruz
from config import PROCESSED_DIR, SEASONS
from constants import UNTOUCHABLE_PLAYERS, TEAM_NAME_TO_ABBR, ABBR_TO_TEAM_NAME

# -----------------------------
# Dosya path'leri ve Çıktı Yolu
# -----------------------------
TEAM_NEEDS_PATH = PROCESSED_DIR / "team_needs.csv"
PLAYER_PROFILES_WEIGHTED_PATH = PROCESSED_DIR / "player_profiles_weighted.csv"

# İstediğin Özel Çıktı Klasörü: data/private/
PROJECT_ROOT = Path(__file__).parent.parent
PRIVATE_OUT_DIR = PROJECT_ROOT / "data" / "private"

DEFAULT_POS_LIMIT = 10
OUT_FILENAME = 'team_player_recs_summer_2024_fixed.csv'

TARGET_TEAM_SEASON = 2025
TARGET_PLAYER_LAST_SEASON = 2024
MIN_GAMES = 0
MIN_MP = 0

# -----------------------------
# Untouchable Players & Team Maps (Aynı bıraktım)
# -----------------------------
UNTOUCHABLE_PLAYERS: Set[str] = {
    "Nikola Jokić", "Victor Wembanyama", "Giannis Antetokounmpo", "Stephen Curry",
    "LeBron James", "Joel Embiid", "Jayson Tatum", "Shai Gilgeous-Alexander",
    "James Harden", "Kyrie Irving", "Kawhi Leonard", "Devin Booker",
    "Donovan Mitchell", "Trae Young", "Ja Morant", "Anthony Edwards",
    "Tyrese Haliburton", "Fred VanVleet", "Nikola Vučević", "Jaylen Brown",
    "Bam Adebayo", "Jamal Murray", "LaMelo Ball", "Chet Holmgren",
    "Tyler Herro", "Jaren Jackson Jr."
 }

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}
ABBR_TO_TEAM_NAME = {abbr: name for name, abbr in TEAM_NAME_TO_ABBR.items()}

# -----------------------------
# Yardımcı Fonksiyonlar & Data Loading (Mantık Aynı)
# -----------------------------

def load_data():
    if not TEAM_NEEDS_PATH.exists() or not PLAYER_PROFILES_WEIGHTED_PATH.exists():
        raise FileNotFoundError("Girdi CSV dosyaları processed klasöründe bulunamadı!")
    return pd.read_csv(TEAM_NEEDS_PATH), pd.read_csv(PLAYER_PROFILES_WEIGHTED_PATH)

def clean_team_name(name: str) -> str:
    return name.replace("*", "").strip() if isinstance(name, str) else ""

def normalize_series_safe(s: pd.Series, min_target: float = 0.1, max_target: float = 1.0) -> pd.Series:
    """Seriyi 0.1 - 1.0 arasına çeker. En kötü oyuncu bile 0 almaz."""
    s_min, s_max = s.min(), s.max()
    if s_max <= s_min:
        return pd.Series(min_target, index=s.index)
    
    # 0..1
    norm = (s - s_min) / (s_max - s_min)
    return min_target + norm * (max_target - min_target)

def prepare_player_scores(df_players: pd.DataFrame) -> pd.DataFrame:
    df = df_players.copy()
    df = df[(df["last_season"] == TARGET_PLAYER_LAST_SEASON) & (df["G"] >= MIN_GAMES) & (df["MP"] >= MIN_MP)]
    
    def col_s(n): return pd.to_numeric(df[n], errors="coerce").fillna(0.0) if n in df.columns else pd.Series(0.0, index=df.index)
    
    df["Defense_Score"] = (col_s("STL")*1.0) + (col_s("BLK")*1.0) + (col_s("DRB")*1.0)
    df["Spacing_Score"] = (col_s("3P")*1.0) + (col_s("3PA")*1.0)
    df["Efficiency_Score"] = (col_s("PTS")*1.0) + (col_s("AST")*1.0) - (col_s("TOV")*1.0)
    
    # YENİ: Star Score (Hacim Bonusu)
    # Alakasız oyuncuları elemek için PTS ve genel katkı hacmini ödüllendiriyoruz.
    rebs = col_s("TRB") if "TRB" in df.columns else col_s("DRB")
    df["Star_Score"] = (col_s("PTS") * 1.0) + (col_s("AST") * 1.0) + (rebs * 1.0)

    # PQ_raw güncellemesi: Star Score'u da katalım (%30 etki)
    df["PQ_raw"] = (df["Efficiency_Score"]*1.0) + (df["Defense_Score"]*1.0) + (df["Spacing_Score"]*1.0) + (df["Star_Score"]*1.0)

    for c in ["Defense_Score", "Spacing_Score", "Efficiency_Score", "PQ_raw"]:
       df[c+"_norm"] = normalize_series_safe(df[c], 0.1, 1.0)
    
    # MP Factor: Dakika süresi arttıkça puanı katla (Yıldızlar çok oynar).
    # MP toplam dakika olduğu için: 1000 dk = 1.0x, 2000 dk = 2.0x
    df["MP_Factor"] = 1
    return df

# -----------------------------
# Yeni İnteraktif Fonksiyon
# -----------------------------

def get_position_limits_from_user(default_limit: int = DEFAULT_POS_LIMIT) -> Dict[str, int]:
    """Terminalden her pozisyon için kaç oyuncu istendiğini sorar."""
    positions = ["PG", "SG", "SF", "PF", "C"]
    limits = {}
    print("\n" + "="*50)
    print("🏀 NBA TAHMİN SİSTEMİ: POZİSYON LİMİTLERİ".center(50))
    print("="*50)
    print("Lütfen her pozisyon için kaçar öneri istediğinizi girin (Enter = varsayılan):")
    
    for pos in positions:
        while True:
            val = input(f"👉 {pos} pozisyonu için limit: ").strip()
            if val == "":
                limits[pos] = int(default_limit)
                break
            if val.isdigit():
                limits[pos] = int(val)
                break
            print("❌ Geçersiz giriş! Lütfen bir sayı girin.")
    return limits

# -----------------------------
# Global Draft & Build (Limitler Sözlük Olarak Alınır)
# -----------------------------

def build_global_recommendations(
    df_team_needs: pd.DataFrame,
    df_players_scored: pd.DataFrame,
    pos_limits: Dict[str, int]
) -> pd.DataFrame:
    rows = []
    untouchables = UNTOUCHABLE_PLAYERS
    
    # 1. Tüm kombinasyonlar için Fit Score hesapla
    for _, trow in df_team_needs.iterrows():
        t_name = clean_team_name(str(trow["team"]))
        t_abbr = trow["team_abbr"]
        if not t_abbr: continue
        
        for _, prow in df_players_scored.iterrows():
            pname = prow["player_name"]
            if pname in untouchables or prow["last_team"] == t_abbr: continue

            pos = str(prow.get("primary_position", "UNK")).upper()
            
            # İSTEK: Basit algoritma (Tüm katsayılar 1)
            fit = (prow["Spacing_Score_norm"] * 1.0 + 
                   prow["Defense_Score_norm"] * 1.0 + 
                   prow["PQ_raw_norm"] * 1.0) * prow["MP_Factor"]
            
            if fit > 0:
                rows.append({
                    "team": t_name, "team_abbr": t_abbr, "target_position": pos,
                    "player_name": pname, "last_team": prow["last_team"], "fit_score": fit
                })

    df_pairs = pd.DataFrame(rows).sort_values("fit_score", ascending=False)
    
    # 2. İnteraktif Limitleri Uygula
    accepted = []
    counts = {} # (team, pos) -> count

    for _, row in df_pairs.iterrows():
        key = (row["team_abbr"], row["target_position"])
        limit = int(pos_limits.get(row["target_position"], DEFAULT_POS_LIMIT))
        
        current_count = counts.get(key, 0)
        if current_count < limit:
            accepted.append(row)
            counts[key] = current_count + 1
            
    df_out = pd.DataFrame(accepted)
    df_out["Rank"] = df_out.groupby(["team_abbr", "target_position"])["fit_score"].rank(ascending=False, method="first").astype(int)
    
    return df_out[["team", "team_abbr", "target_position", "Rank", "player_name", "last_team", "fit_score"]]

# -----------------------------
# Main Execution
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["draft", "needs", "recs"], default="draft")
    parser.add_argument("--team", type=str)
    parser.add_argument("--interactive", action="store_true", help="Pozisyon limitlerini terminalden sor (pipeline için kapalı tut).")
    parser.add_argument("--default-limit", type=int, default=DEFAULT_POS_LIMIT, help="Non-interaktif modda her pozisyon için varsayılan öneri sayısı.")
    parser.add_argument("--also-private-copy", action="store_true", help="Çıktıyı data/private altına da kopyala.")
    args = parser.parse_args()

    # Veriyi yükle ve skorla
    df_team_needs, df_p_raw = load_data()
    df_players_scored = prepare_player_scores(df_p_raw)

    if args.mode == "draft":
        # Pipeline/otomasyon çalıştırmalarında stdin yoksa input() patlar.
        # Bu yüzden non-interaktif modda varsayılan limitlerle devam eder.
        if args.interactive or sys.stdin.isatty():
            user_limits = get_position_limits_from_user(default_limit=int(args.default_limit))
        else:
            user_limits = {p: int(args.default_limit) for p in ["PG", "SG", "SF", "PF", "C"]}
            print(f"[INFO] Non-interaktif çalıştırma: pozisyon limitleri varsayılan={args.default_limit}")

        # HESAPLA
        print("\n⚙️  Draft hesaplanıyor, lütfen bekleyin...")
        df_recs = build_global_recommendations(df_team_needs, df_players_scored, user_limits)
        
        # KAYDET (kanonik çıktı: data/processed/)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / OUT_FILENAME
        df_recs.to_csv(out_path, index=False)

        if args.also_private_copy:
            PRIVATE_OUT_DIR.mkdir(parents=True, exist_ok=True)
            df_recs.to_csv(PRIVATE_OUT_DIR / OUT_FILENAME, index=False)

        print(f"\n✅ Başarılı! Öneriler kaydedildi: {out_path}")
        print(f"📊 Toplam Öneri Sayısı: {len(df_recs)}")

    elif args.mode == "needs":
        # Mevcut ihtiyaç gösterme mantığı (opsiyonel)
        print("İhtiyaç analizi modu aktif.")

if __name__ == "__main__":
    main()