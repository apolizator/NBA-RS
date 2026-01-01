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
import subprocess
import re

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
UNTOUCHABLE_PLAYERS: Set[str] = { }

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
    
    df["Defense_Score"] = (col_s("STL")*2) + (col_s("BLK")*1.5) + (col_s("DRB")*0.5)
    df["Spacing_Score"] = (col_s("3P")*1.5) + (col_s("3PA")*0.5)
    df["Efficiency_Score"] = (col_s("PTS")*1) + (col_s("AST")*1.5) - (col_s("TOV")*2)
    
    # YENİ: Star Score (Hacim Bonusu)
    # Alakasız oyuncuları elemek için PTS ve genel katkı hacmini ödüllendiriyoruz.
    rebs = col_s("TRB") if "TRB" in df.columns else col_s("DRB")
    df["Star_Score"] = (col_s("PTS") * 2.0) + (col_s("AST") * 1.0) + (rebs * 1.0)

    # PQ_raw güncellemesi: Star Score'u da katalım (%30 etki)
    df["PQ_raw"] = (df["Efficiency_Score"]*0.3) + (df["Defense_Score"]*0.2) + (df["Spacing_Score"]*0.2) + (df["Star_Score"]*0.3)

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
            
            # İSTEK: Tamamen rastgele skorlama (Chaos Mode)
            fit = random.uniform(0.1, 10.0)
            
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
# Simulation Mode
# -----------------------------

def run_simulation(df_team_needs: pd.DataFrame, df_players_scored: pd.DataFrame, pos_limits: Dict[str, int], iterations: int):
    """Simülasyon modu: Rastgele önerileri tekrar tekrar üretip f.py ile skorlar."""
    
    # f.py dosyasını ara (aynı dizinde veya bir üstte)
    current_dir = Path(__file__).parent
    f_path = current_dir / "f.py"
    if not f_path.exists():
        f_path = current_dir.parent / "f.py"
    
    if not f_path.exists():
        print(f"❌ Hata: 'f.py' dosyası {current_dir} veya üst dizininde bulunamadı.")
        return

    print(f"\n🚀 Simülasyon Başlatılıyor... ({iterations} iterasyon)")
    print(f"📄 Kontrol Scripti: {f_path}")
    
    hits_list = []
    out_path = PROCESSED_DIR / OUT_FILENAME
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, iterations + 1):
        # Önerileri oluştur
        df_recs = build_global_recommendations(df_team_needs, df_players_scored, pos_limits)
        df_recs.to_csv(out_path, index=False)
        
        # f.py çalıştır
        try:
            result = subprocess.run([sys.executable, str(f_path)], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"   [HATA] f.py çalışırken hata oluştu: {result.stderr.strip()}")
                hits = 0
            else:
                output = result.stdout.strip()
                # Regex düzeltildi: "Total Hits" (boşluklu) yakalanmalı
                match = re.search(r"Total\s+Hits\s*[:=]?\s*(\d+)", output, re.IGNORECASE)
                if match:
                    hits = int(match.group(1))
                else:
                    hits = 0
            
            hits_list.append(hits)
            avg_hits = sum(hits_list) / len(hits_list)
            print(f"   Iter {i:03d}: Gelen Hits={hits} | ANLIK ORTALAMA={avg_hits:.4f}")

        except Exception as e:
            print(f"   Iter {i:03d}: Hata -> {e}")

    if hits_list:
        final_avg_hits = sum(hits_list) / len(hits_list)
        print("\n" + "="*40)
        print(f"📊 ORTALAMA TOTAL HITS: {final_avg_hits:.4f}")
        print("="*40)

# -----------------------------
# Main Execution
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["draft", "needs", "recs", "simulate"], default="draft")
    parser.add_argument("--team", type=str)
    parser.add_argument("--interactive", action="store_true", help="Pozisyon limitlerini terminalden sor (pipeline için kapalı tut).")
    parser.add_argument("--default-limit", type=int, default=DEFAULT_POS_LIMIT, help="Non-interaktif modda her pozisyon için varsayılan öneri sayısı.")
    parser.add_argument("--also-private-copy", action="store_true", help="Çıktıyı data/private altına da kopyala.")
    parser.add_argument("--iterations", type=int, default=10, help="Simülasyon tekrar sayısı.")
    args = parser.parse_args()

    # Veriyi yükle ve skorla
    df_team_needs, df_p_raw = load_data()
    df_players_scored = prepare_player_scores(df_p_raw)

    # Filtrelenen oyuncuları raporla
    filtered_logs = []
    raw_season_players = set(df_p_raw[df_p_raw["last_season"] == TARGET_PLAYER_LAST_SEASON]["player_name"].unique())
    scored_players = set(df_players_scored["player_name"].unique())

    # 1. Süre/Maç nedeniyle elenenler (Raw datada var ama Scored datada yok)
    for p in (raw_season_players - scored_players):
        filtered_logs.append({"player_name": p, "reason": "Time/Game Filter"})

    # 2. Untouchable olduğu için elenenler (Scored datada var ama Untouchable listesinde)
    for p in scored_players:
        if p in UNTOUCHABLE_PLAYERS:
            filtered_logs.append({"player_name": p, "reason": "Untouchable"})

    if filtered_logs:
        pd.DataFrame(filtered_logs).to_csv(PROCESSED_DIR / "filtered_players_log.csv", index=False)

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

    elif args.mode == "simulate":
        # Limitleri sor
        if args.interactive or sys.stdin.isatty():
            user_limits = get_position_limits_from_user(default_limit=int(args.default_limit))
            try:
                val = input(f"Kaç kez çalıştırılsın? (Varsayılan: {args.iterations}): ").strip()
                iterations = int(val) if val.isdigit() else args.iterations
            except:
                iterations = args.iterations
        else:
            user_limits = {p: int(args.default_limit) for p in ["PG", "SG", "SF", "PF", "C"]}
            iterations = args.iterations
        
        run_simulation(df_team_needs, df_players_scored, user_limits, iterations)

    elif args.mode == "needs":
        # Mevcut ihtiyaç gösterme mantığı (opsiyonel)
        print("İhtiyaç analizi modu aktif.")

if __name__ == "__main__":
    main()