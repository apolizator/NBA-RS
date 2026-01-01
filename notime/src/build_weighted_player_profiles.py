"""
build_weighted_player_profiles.py

Amaç:
- data/processed/players_stats_master.csv dosyasından
  her oyuncu için TEK SATIRLIK, çok sezonlu ve pozisyondan bağımsız
  ağırlıklı istatistik profili üretmek.

Özellikler:
- group_by: player_name  (pozisyona göre AYRILMIYOR, tek oyuncu)
- Sezonlar SEASONS listesine göre ağırlıklandırılır:
    en eski sezona 1, sonra 2, ... en yeni sezona N
- Her oyuncu için:
    * player_name
    * primary_position  : son sezonda / en sık oynadığı pozisyon
    * positions_all     : oynadığı tüm pozisyonlar (örn: "PG/SG/SF")
    * last_team         : son sezondaki gerçek takım (mümkünse TOT olmayan)
    * last_season       : eldeki en yeni sezon
    * seasons_count     : kaç farklı sezon verisi var
    * weighted stat ortalamaları (PTS, TRB, AST, 2P, 2P%, 3P, 3P%, TS%, vs.)

Filtre:
- YALNIZCA last_season == TARGET_PLAYER_LAST_SEASON (2024) olan oyuncular
  CSV'de kalır. Diğerleri tamamen elenir.

Çıktı:
- data/processed/player_profiles_weighted.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import PROCESSED_DIR, SEASONS


PLAYERS_MASTER_PATH = PROCESSED_DIR / "players_stats_master.csv"
PLAYER_PROFILES_WEIGHTED_PATH = PROCESSED_DIR / "player_profiles_weighted.csv"

# Sadece son sezonu 2024 olan oyuncular tutulacak
TARGET_PLAYER_LAST_SEASON = 2024


# -------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# -------------------------------------------------------

def load_players_master() -> pd.DataFrame:
    """
    Çok sezonlu oyuncu istatistiklerini yükler.

    Beklenen minimum kolonlar:
        - season
        - player_name
    Varsa:
        - position
        - team
        - G, MP, PTS, TRB, AST, 2P, 2P%, 3P, 3P%, TS%, ...
    """
    if not PLAYERS_MASTER_PATH.exists():
        raise FileNotFoundError(f"Master oyuncu dosyası yok: {PLAYERS_MASTER_PATH}")

    df = pd.read_csv(PLAYERS_MASTER_PATH)

    required_cols = ["season", "player_name"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Beklenen kolon eksik: '{col}'. Mevcut kolonlar: {list(df.columns)}"
            )

    return df


def build_season_weights() -> Dict[int, float]:
    """
    SEASONS listesindeki sezonlara göre ağırlık sözlüğü üretir.

    Mantık:
        en eski sezona 1,
        bir sonrakine 2,
        ...
        en yeni sezona N
    """
    seasons_sorted = sorted(SEASONS)
    weights: Dict[int, float] = {}

    for idx, season in enumerate(seasons_sorted, start=1):
        weights[season] = float(idx)

    return weights


def choose_last_team_and_positions(group: pd.DataFrame) -> Tuple[str, str, str, int]:
    """
    Bir oyuncu grubundan:
    - last_season: elimizdeki en son sezon
    - last_team  : o sezondaki gerçek takım (mümkünse 'TOT' olmayan satır)
    - primary_position : son sezonda / genel olarak en sık görülen pozisyon
    - positions_all    : tüm sezondaki pozisyonların birleşimi ("PG/SG/SF" gibi)
    """
    seasons_for_player = group["season"].dropna().astype(int).unique().tolist()
    if not seasons_for_player:
        return None, None, None, None  # type: ignore

    last_season = int(max(seasons_for_player))

    # --- last_team ve primary_position (son sezona odaklı) ---
    last_season_rows = group[group["season"] == last_season].copy()

    last_team = None
    primary_position = None

    # Takım seçimi (Hangi kolonun olduğunu tespit et)
    team_col = "team_abbr" if "team_abbr" in last_season_rows.columns else "team"

    if team_col in last_season_rows.columns and not last_season_rows.empty:
        # Önce TOT olmayan satırlara bak
        non_tot = last_season_rows[last_season_rows[team_col] != "TOT"]
        if not non_tot.empty:
            # En çok maça çıkılan satırı al (G kolonu varsa)
            if "G" in non_tot.columns:
                non_tot = non_tot.sort_values("G", ascending=False)
            row = non_tot.iloc[0]
        else:
            # Sadece TOT varsa onu al
            row = last_season_rows.iloc[0]

        last_team = row.get(team_col, None)


    # Primary position seçimi
    if "position" in last_season_rows.columns and not last_season_rows.empty:
        pos_col = last_season_rows["position"].dropna().astype(str)
        if not pos_col.empty:
            # En sık görüleni al
            primary_position = pos_col.mode().iloc[0]

    # Eğer son sezonda pozisyon boşsa, tüm grup üzerinden en sık görüleni al
    if primary_position is None and "position" in group.columns:
        pos_col_all = group["position"].dropna().astype(str)
        if not pos_col_all.empty:
            primary_position = pos_col_all.mode().iloc[0]

    # --- positions_all: tüm sezondaki pozisyonların birleşimi ---
    positions_all = None
    if "position" in group.columns:
        pos_all = group["position"].dropna().astype(str).unique().tolist()
        if pos_all:
            # Tekrarsız ve sıralı gösterim
            pos_all_sorted = sorted(pos_all)
            positions_all = "/".join(pos_all_sorted)

    if last_team is not None and not (isinstance(last_team, float) and pd.isna(last_team)):
        last_team = str(last_team).upper().strip()
    else:
        last_team = None

    return last_team, primary_position, positions_all, last_season


# -------------------------------------------------------
# ANA İŞLEV
# -------------------------------------------------------

def build_weighted_profiles() -> Path:
    """
    players_stats_master.csv'den:
    - player_name başına tek satır olmak üzere
    - sezon ağırlıklı oyuncu profilleri üretir ve kaydeder.
    - Son sezonu TARGET_PLAYER_LAST_SEASON (2024) olmayan oyuncuları ELER.
    """
    df = load_players_master()

    # Sadece SEASONS listesindeki sezonlar
    df = df[df["season"].isin(SEASONS)].copy()
    if df.empty:
        raise ValueError(
            f"SEASONS={SEASONS} için players_stats_master içinde satır bulunamadı."
        )

    # Sezon ağırlıkları
    season_weights = build_season_weights()

    # Numeric kolonlar (season hariç)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "season" in num_cols:
        num_cols.remove("season")

    if not num_cols:
        raise ValueError("Kullanılabilir numeric istatistik kolonu bulunamadı.")

    group_col = "player_name"

    profiles: List[dict] = []

    # Oyuncu bazlı grupla
    for player_name, group in df.groupby(group_col):
        group = group.copy()

        # Sezonlar
        seasons_for_player = (
            group["season"].dropna().astype(int).unique().tolist()
        )
        if not seasons_for_player:
            continue

        # Bu oyuncu için ilgili sezon ağırlıkları
        local_weights = {s: season_weights.get(s, 0.0) for s in seasons_for_player}
        local_weights = {s: w for s, w in local_weights.items() if w > 0.0}
        if not local_weights:
            continue

        total_w = sum(local_weights.values())

        # Her satıra weight kolonu
        group["w"] = group["season"].map(local_weights).fillna(0.0)

        # Weighted ortalama: sum(w * x) / sum(w)
        weighted_vals = {}
        for col in num_cols:
            vals = group[col].fillna(0.0)
            w = group["w"]
            num = (vals * w).sum()
            if total_w == 0.0:
                weighted_vals[col] = 0.0
            else:
                weighted_vals[col] = num / total_w

        # Son takım, primary_position, positions_all ve last_season
        last_team, primary_position, positions_all, last_season = \
            choose_last_team_and_positions(group)

        profile = {
            "player_name": player_name,
            "primary_position": primary_position,
            "positions_all": positions_all,
            "last_team": last_team,
            "last_season": last_season,
            "seasons_count": len(seasons_for_player),
        }

        profile.update(weighted_vals)
        profiles.append(profile)

    if not profiles:
        raise ValueError("Hiç oyuncu profili üretilemedi.")

    df_profiles = pd.DataFrame(profiles)

    # Float kolonları 4 haneye yuvarla
    float_cols = df_profiles.select_dtypes(include=["float64", "float32"]).columns
    df_profiles[float_cols] = df_profiles[float_cols].round(4)

    # --- SON SEZONU 2024 OLMAYAN OYUNCULARI ELE ---
    if "last_season" in df_profiles.columns:
        before = len(df_profiles)
        df_profiles = df_profiles[df_profiles["last_season"] == TARGET_PLAYER_LAST_SEASON]
        after = len(df_profiles)
        print(
            f"[INFO] last_season == {TARGET_PLAYER_LAST_SEASON} filtresi uygulandı: "
            f"{before} -> {after} oyuncu"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_profiles.to_csv(PLAYER_PROFILES_WEIGHTED_PATH, index=False)

    print(f"[INFO] Ağırlıklı oyuncu profilleri kaydedildi: {PLAYER_PROFILES_WEIGHTED_PATH}")
    print("[INFO] Her oyuncu player_name bazında TEK satırdır (pozisyondan bağımsız).")
    print("[INFO] Pozisyon bilgileri: primary_position + positions_all alanlarında tutulur.")
    print(f"[INFO] Sadece last_season == {TARGET_PLAYER_LAST_SEASON} olan oyuncular dosyada tutuldu.")

    return PLAYER_PROFILES_WEIGHTED_PATH


def main() -> None:
    build_weighted_profiles()


if __name__ == "__main__":
    main()