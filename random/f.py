#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_check_all_in_one.py (FIXED & COMPLETE)

Amaç:
- Pipeline çıktılarını analiz eder.
- "Genel Başarı Tablosu" (Mutual, Masterclass sayıları) GERİ EKLENDİ.
- Terminalin en altında algoritmanın TOPLAM PUANINI yazar.

Kullanım:
  python3 f.py --mode all --processed-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np
import unicodedata
try:
    import ftfy
except ImportError:
    ftfy = None


# -----------------------------
# Basit logger
# -----------------------------
def log(msg: str = "") -> None:
    print(msg)

# -----------------------------
# İsim Temizleme (Ortak)
# -----------------------------
def clean_player_name(x: str) -> str:
    x = str(x)
    if ftfy:
        x = ftfy.fix_text(x)
    x = unicodedata.normalize("NFC", x)
    return x.replace("*", "").replace("\u00a0", " ").strip()

# -----------------------------
# Kolon tespiti (ortak)
# -----------------------------
def detect_player_col(df: pd.DataFrame) -> str:
    candidates = ["player_name", "Player", "PLAYER", "Name", "name"]
    for c in candidates:
        if c in df.columns: return c
    raise ValueError("Player name column not found.")

def detect_team_col(df: pd.DataFrame) -> Tuple[str, bool]:
    abbr_candidates = ["team_abbr", "Tm", "tm", "Team", "team"]
    for c in abbr_candidates:
        if c in df.columns: return c, True
    name_candidates = ["Franchise", "TeamName", "TEAM_FULL"]
    for c in name_candidates:
        if c in df.columns: return c, False
    raise ValueError("Team column not found.")

def detect_pos_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["target_position", "position", "pos", "Pos", "primary_position", "player_position"]:
        if c in df.columns: return c
    return None

# -----------------------------
# Dosya çözümleme / yükleme
# -----------------------------
def resolve_default_processed_dir(arg_dir: str) -> Path:
    if arg_dir: return Path(arg_dir)
    p1 = Path("data/processed")
    if p1.exists(): return p1
    return Path(".")

def load_recommendations(recs_path: Path) -> pd.DataFrame:
    if not recs_path.exists():
        raise FileNotFoundError(f"Recommendation file not found: {recs_path}")
    df = pd.read_csv(recs_path)
    df = df.copy()
    df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce")
    df = df.dropna(subset=["fit_score"])
    
    # Rank yoksa oluştur
    if "Rank" not in df.columns:
        df = df.sort_values(["team_abbr", "fit_score"], ascending=[True, False])
        df["Rank"] = df.groupby("team_abbr").cumcount() + 1
        
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].apply(clean_player_name)
        
    return df

def load_active_rosters(active_path: Path) -> pd.DataFrame:
    if not active_path.exists():
        raise FileNotFoundError(f"Active roster file not found: {active_path}")
    df = pd.read_csv(active_path)
    player_col = detect_player_col(df)
    if player_col != "player_name": df = df.rename(columns={player_col: "player_name"})
    team_col, is_abbr = detect_team_col(df)
    if is_abbr:
        if team_col != "team_abbr": df = df.rename(columns={team_col: "team_abbr"})
    else:
        if team_col != "team_full": df = df.rename(columns={team_col: "team_full"})
        # If only team_full exists, use it as team_abbr
        if "team_abbr" not in df.columns:
            df["team_abbr"] = df["team_full"]
    
    df = df.drop_duplicates(subset=["player_name"], keep="first")
    df["player_name"] = df["player_name"].apply(clean_player_name)
    return df[["player_name", "team_abbr"]].copy()


# ============================================================
# 1) PLAYER_EVAL
# ============================================================
def evaluate_by_player(recs_path: Path, active_path: Path, max_k: int = 10) -> None:
    log("📂 Loading files...")
    df_recs = load_recommendations(recs_path)
    df_act = load_active_rosters(active_path)
    player_to_team = dict(zip(df_act["player_name"], df_act["team_abbr"]))
    grouped = df_recs.groupby("player_name", dropna=True)

    total_players = 0
    top1_hits = top3_hits = top5_hits = 0
    sample_top3 = []

    for player_name, recs in grouped:
        actual_team = player_to_team.get(player_name)
        if not actual_team: continue
        recs_sorted = recs.sort_values("fit_score", ascending=False).drop_duplicates(subset=["team_abbr"], keep="first")
        candidate_teams = recs_sorted["team_abbr"].tolist()
        if not candidate_teams: continue
        total_players += 1
        try: rank = candidate_teams.index(actual_team) + 1
        except ValueError: rank = None

        if rank == 1: top1_hits += 1
        if rank and rank <= 3: 
            top3_hits += 1
            if len(sample_top3) < 5:
                sample_top3.append({"player": player_name, "team": actual_team, "rank": rank})
        if rank and rank <= 5: top5_hits += 1

    def pct(x): return (100.0 * x / total_players) if total_players else 0.0
    log("\n=== PLAYER-BASED EVALUATION ===")
    log(f"Top-1: {top1_hits} ({pct(top1_hits):.1f}%) | Top-3: {top3_hits} ({pct(top3_hits):.1f}%) | Top-5: {top5_hits} ({pct(top5_hits):.1f}%)")


# ============================================================
# 2) TEAM_RECS
# ============================================================
def evaluate_team_recommendations(recs_path: Path, active_path: Optional[Path], topk: int = 10, save_csv: Optional[Path] = None) -> None:
    df_recs = load_recommendations(recs_path)
    player_to_team = {}
    if active_path and active_path.exists():
        df_act = load_active_rosters(active_path)
        player_to_team = dict(zip(df_act["player_name"], df_act["team_abbr"]))

    teams = sorted(df_recs["team_abbr"].unique())
    summary_rows = []

    for t in teams:
        df_t = df_recs[df_recs["team_abbr"] == t].sort_values("fit_score", ascending=False).drop_duplicates(subset=["player_name"], keep="first")
        df_top = df_t.head(topk).copy()
        df_top["actual_team"] = df_top["player_name"].map(player_to_team).fillna("UNK")
        
        avg_fit = float(df_top["fit_score"].mean()) if len(df_top) else 0.0
        summary_rows.append({"Team": t, f"top{topk}_avg_fit": avg_fit})

    df_sum = pd.DataFrame(summary_rows).sort_values(f"top{topk}_avg_fit", ascending=False)
    log("\n=== AVERAGE RECOMMENDATION QUALITY PER TEAM (FIT SCORE) ===")
    log(df_sum.head(10).to_string(index=False))
    if save_csv:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(save_csv, index=False)


# ============================================================
# 3) HITS_BY_POS (Mutual Matches & Success Table)
# ============================================================
def get_success_label(rank: int, is_mutual: bool) -> str:
    if rank == 1: return "🎯 MASTERCLASS"
    elif rank <= 5: return "🔥 ELITE PICK"
    elif rank <= 10: return "✅ STRONG PREDICTION"
    else: return "🟡 HIT"

def evaluate_hits_by_pos(recs_path: Path, active_path: Path, out_csv_path: Path, top_player_k: int = 5) -> None:
    df_recs = load_recommendations(recs_path)
    df_act = load_active_rosters(active_path)
    player_to_team = dict(zip(df_act["player_name"], df_act["team_abbr"]))

    # Player Rank Hesapla
    if "PlayerTeamRank" not in df_recs.columns:
        df_recs["PlayerTeamRank"] = df_recs.groupby("player_name")["fit_score"].rank(ascending=False, method="first")
    
    player_top_fits = {}
    for p_name in df_recs["player_name"].dropna().unique():
        p_data = df_recs[df_recs["player_name"] == p_name].sort_values("fit_score", ascending=False)
        player_top_fits[p_name] = p_data.head(top_player_k)["team_abbr"].tolist()

    hits_list = []
    summary_rows = []
    teams = sorted(df_recs["team_abbr"].dropna().unique())

    for team in teams:
        df_team = df_recs[df_recs["team_abbr"] == team]
        team_hits = []
        for _, row in df_team.iterrows():
            p_name = row["player_name"]
            if player_to_team.get(p_name) == team:
                is_mutual = (int(row["Rank"]) <= top_player_k and team in player_top_fits.get(p_name, []))
                status = get_success_label(int(row["Rank"]), is_mutual)
                
                entry = {
                    "Team": team, "Player": p_name, "Rank": int(row["Rank"]), 
                    "Fit": float(row["fit_score"]), "Mutual": is_mutual, "Status": status
                }
                team_hits.append(entry)
                hits_list.append(entry)
        
        # Takım özeti (Success Table için gerekli)
        if team_hits:
            mutual_count = sum(1 for h in team_hits if h["Mutual"])
            master_count = sum(1 for h in team_hits if "MASTERCLASS" in h["Status"])
            elite_count = sum(1 for h in team_hits if "ELITE" in h["Status"])
            
            summary_rows.append({
                "Team": team,
                "🤝 Mutual Match": mutual_count,
                "🎯 Masterclass": master_count,
                "🔥 Elite": elite_count,
                "Total_Hits": len(team_hits)
            })

    if hits_list:
        df_hits = pd.DataFrame(hits_list).sort_values(["Rank", "Fit"], ascending=[True, False])
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_hits.to_csv(out_csv_path, index=False)

        # --- GENEL BAŞARI TABLOSU ---
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            
            # Detaylı sayım
            total_master = sum(1 for h in hits_list if "MASTERCLASS" in h["Status"])
            total_elite = sum(1 for h in hits_list if "ELITE" in h["Status"])
            total_strong = sum(1 for h in hits_list if "STRONG" in h["Status"])
            total_good = sum(1 for h in hits_list if "HIT" in h["Status"])
            total_hits = len(hits_list)

            log("\n" + "📊 GENERAL SUCCESS TABLE")
            log("─"*40)
            log(f"🎯 Masterclass     : {int(total_master)}")
            log(f"🔥 Elite Picks     : {int(total_elite)}")
            log(f"✅ Strong Predictions: {int(total_strong)}")
            log(f"🟡 Good Attempts     : {int(total_good)}")
            log("─"*40)
            log(f"✅ Total Hits        : {total_hits}")
            log("─"*40)

            # --- ŞAMPİYON TAKIMLAR ---
            df_sum = df_sum.sort_values("Total_Hits", ascending=False)
            log("\n🏆 PREDICTION CHAMPION TEAMS:")
            for i, (_, r) in enumerate(df_sum.head(3).iterrows(), 1):
                log(f"{i}. {r['Team']} ({r['Total_Hits']} Correct)")


# ============================================================
# 4) SCORE CALCULATION (EN ALTTA GÖRÜNECEK KISIM)
# ============================================================
def print_total_algorithm_score(recs_path: Path, active_path: Path, top_k: int = 100) -> None:
    """
    Algoritmanın nihai başarısını tek bir puanla özetler.
    Puanlama:
      - Rank 1: 100 Puan
      - Rank 2-5: 50 Puan
      - Rank 6-10: 25 Puan
      - Rank > 10: 10 Puan
    """
    df_recs = load_recommendations(recs_path)
    df_act = load_active_rosters(active_path)
    
    real_rosters = df_act.groupby("team_abbr")["player_name"].apply(set).to_dict()
    teams = df_recs["team_abbr"].unique()
    
    total_score = 0
    total_hits = 0
    
    for team in teams:
        if team not in real_rosters: continue
        real_players = real_rosters[team]
        recs = df_recs[(df_recs["team_abbr"] == team) & (df_recs["Rank"] <= top_k)]
        
        for _, row in recs.iterrows():
            if row["player_name"] in real_players:
                rank = int(row["Rank"])
                if rank == 1:
                    points = 100
                elif 2 <= rank <= 5:
                    points = 50
                elif 6 <= rank <= 10:
                    points = 25
                else:
                    points = 10
                
                total_score += points
                total_hits += 1
    
    avg_score = total_score / len(teams) if len(teams) > 0 else 0
    
    log("\n" + "█"*60)
    log(f"📊  ALGORITHM PERFORMANCE REPORT (TOP-{top_k})  📊".center(60))
    log("█"*60)
    log(f"   ➤ TOTAL SCORE (Weighted Score):  {int(total_score)}")
    log(f"   ➤ TOTAL PLAYERS CAUGHT:          {total_hits}")
    log(f"   ➤ AVERAGE SCORE PER TEAM:        {avg_score:.2f}")
    log("█"*60 + "\n")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Final Check All-in-One")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "score", "hits", "eval"], help="Mode selection")
    parser.add_argument("--processed-dir", type=str, default="", help="processed folder")
    parser.add_argument("--active", type=str, default="", help="Active roster CSV")
    parser.add_argument("--recs", type=str, default="", help="Recommendation CSV")
    
    args = parser.parse_args()

    processed_dir = resolve_default_processed_dir(args.processed_dir)
    recs_path = Path(args.recs) if args.recs else (processed_dir / "team_player_recs_summer_2024_fixed.csv")
    active_path = Path(args.active) if args.active else (processed_dir / "aktifnba2025-26.csv")
    
    team_recs_out = processed_dir / "final_check_team_recs.csv"
    hits_out = processed_dir / "final_basari_analizi.csv"

    try:
        # 1. Modül: Oyuncu Değerlendirme
        if args.mode in ("all", "eval"):
            evaluate_by_player(recs_path, active_path)
            evaluate_team_recommendations(recs_path, active_path, save_csv=team_recs_out)

        # 2. Modül: Mutual Matches ve Başarı Tablosu
        if args.mode in ("all", "hits"):
            evaluate_hits_by_pos(recs_path, active_path, hits_out)

        # 3. Modül: EN SON ÇALIŞACAK VE PUANI YAZACAK
        if args.mode in ("all", "score"):
            print_total_algorithm_score(recs_path, active_path, top_k=100)

    except FileNotFoundError as e:
        log(f"❌ File Not Found: {e}")
    except Exception as e:
        log(f"❌ Error: {e}")

if __name__ == "__main__":
    main()