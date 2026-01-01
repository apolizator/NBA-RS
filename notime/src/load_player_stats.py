# -*- coding: utf-8 -*-
"""
src/load_player_stats.py

Amaç:
- data/raw altında bulunan sezonluk player_stats_YYYY_YY.csv dosyalarını okuyup tek master CSV üretmek.

Multi-team düzeltmesi (İSTENEN):
- Aynı oyuncu aynı sezonda birden fazla takımda oynadıysa BBRef export genelde:
  - Team = 2TM/3TM veya TOT (toplam satır)
  - Team = ilk takım
  - Team = ikinci takım
  şeklinde satırlar içerir.

Bu projede istenen davranış:
- Eğer oyuncunun aynı sezonda 2TM/TOT satırı VARSA:
    * SADECE 2TM/TOT satırını tut
    * Diğer gerçek takım satırlarını sil
    * 2TM/TOT satırının team değerini, oyuncunun o sezondaki İLK gerçek takımına çevir
- Edge-case (2TM/TOT yok ama 2+ gerçek takım satırı var):
    * Gerçek takım satırlarının numeric kolonlarının ORTALAMASI alınır
    * Team = ilk takım

Çıktı:
- data/processed/players_stats_master.csv
"""

from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd

from config import RAW_DIR, PROCESSED_DIR


SEASONS: List[str] = [
    "2019_20",
    "2020_21",
    "2021_22",
    "2022_23",
    "2023_24",
]

PLAYER_COL_CANDIDATES = ["Player", "player", "PLAYER", "player_name", "PLAYER_NAME"]
TEAM_COL_CANDIDATES   = ["Team", "team", "TEAM", "Tm", "tm", "team_abbr"]
POS_COL_CANDIDATES    = ["Pos", "pos", "POS", "position", "primary_position"]
_MULTI_TEAM_RE = re.compile(r"^\d+TM$", re.IGNORECASE)  # 2TM, 3TM, ...


def pick_col(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"[load_player_stats] '{label}' kolonu bulunamadı. Mevcut kolonlar: {list(df.columns)}")


def clean_player_name(x: str) -> str:
    return str(x).replace("*", "").replace("\u00a0", " ").strip()


def is_multi_team_label(team_val: str) -> bool:
    t = str(team_val).strip().upper()
    return t == "TOT" or bool(_MULTI_TEAM_RE.match(t))


def to_numeric_where_possible(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    """Meta kolonlar hariç mümkün olan kolonları numeric'e çevirir (sessizce dener)."""
    df = df.copy()
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df


def collapse_multi_team_rows(df: pd.DataFrame, player_col: str, team_col: str, pos_col: str) -> pd.DataFrame:
    """Aynı oyuncu aynı sezonda birden fazla takımda oynadıysa veriyi TEK satıra indir.

    İSTENEN DAVRANIŞ (net):
      - 2TM/3TM/TOT satırı VARSA:
          * SADECE bu satırı tut.
          * Diğer gerçek takım satırlarını (TOR/IND gibi) SİL.
          * 2TM/TOT satırındaki takım etiketini, oyuncunun o sezondaki İLK gerçek takımıyla değiştir.
      - 2TM/TOT satırı YOKSA ama 2+ gerçek takım satırı varsa (edge-case):
          * Gerçek takım satırlarının sayısal kolonlarının ORTALAMASINI al,
          * Takım etiketi yine ilk takım olsun.
    """
    if df.empty:
        return df

    df = df.copy()
    df["__rid"] = range(len(df))  # dosya sırasını korumak için

    # normalize
    df[player_col] = df[player_col].map(clean_player_name)
    df[team_col] = df[team_col].astype(str).str.upper().str.strip()
    df[pos_col] = df[pos_col].astype(str).str.upper().str.strip()

    # numeric dönüşüm (sadece edge-case ortalama için)
    meta_exclude = [player_col, team_col, pos_col, "__rid"]
    df = to_numeric_where_possible(df, exclude=meta_exclude)

    out_rows = []

    for player, g in df.groupby(player_col, sort=False):
        g = g.sort_values("__rid")

        multi = g[g[team_col].map(is_multi_team_label)].copy()
        real  = g[~g[team_col].map(is_multi_team_label)].copy()

        first_team = None
        first_pos = None
        if not real.empty:
            first_team = str(real.iloc[0][team_col]).upper().strip()
            first_pos  = str(real.iloc[0][pos_col]).upper().strip()

        # 1) Asıl istenen: multi satırı varsa sadece onu tut, team'i ilk gerçek takıma çevir
        if not multi.empty and len(real) >= 2:
            multi_sorted = multi.sort_values("__rid")
            tot_rows = multi_sorted[multi_sorted[team_col] == "TOT"]
            base = (tot_rows.iloc[0] if not tot_rows.empty else multi_sorted.iloc[0]).copy()

            if first_team:
                base[team_col] = first_team
            if first_pos:
                base[pos_col] = first_pos

            out_rows.append(base)
            continue

        # 2) Tek gerçek takım varsa onu kullan
        if len(real) == 1:
            out_rows.append(real.iloc[0])
            continue

        # 3) Sadece multi satırı varsa (nadir) onu kullan
        if real.empty and not multi.empty:
            out_rows.append(multi.sort_values("__rid").iloc[0].copy())
            continue

        # 4) Edge-case: multi yok ama 2+ gerçek takım var -> ortalama al
        if len(real) >= 2:
            real = real.sort_values("__rid")
            first_team = str(real.iloc[0][team_col]).upper().strip()
            first_pos  = str(real.iloc[0][pos_col]).upper().strip()

            num_cols = real.select_dtypes(include="number").columns.tolist()
            if "__rid" in num_cols:
                num_cols.remove("__rid")

            base = real.iloc[0].copy()
            base[num_cols] = real[num_cols].mean(numeric_only=True)
            base[team_col] = first_team
            base[pos_col]  = first_pos
            out_rows.append(base)
            continue

        # 5) fallback
        out_rows.append(g.iloc[0])

    out = pd.DataFrame(out_rows).drop(columns=["__rid"], errors="ignore")
    return out


def season_end_year(season_tag: str) -> int:
    # "2023_24" -> 2024
    start = int(season_tag.split("_")[0])
    return start + 1


def load_one_season(season_tag: str) -> pd.DataFrame:
    path = RAW_DIR / f"player_stats_{season_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"[load_player_stats] Sezon CSV yok: {path}")

    df = pd.read_csv(path)

    player_col = pick_col(df, PLAYER_COL_CANDIDATES, "player")
    team_col   = pick_col(df, TEAM_COL_CANDIDATES, "team")
    pos_col    = pick_col(df, POS_COL_CANDIDATES, "pos")

    df = collapse_multi_team_rows(df, player_col, team_col, pos_col)

    df = df.rename(columns={player_col: "player_name", team_col: "team_abbr", pos_col: "position"})

    df["player_name"] = df["player_name"].map(clean_player_name)
    df["team_abbr"]   = df["team_abbr"].astype(str).str.upper().str.strip()
    df["position"]    = df["position"].astype(str).str.upper().str.strip()
    df["season"]      = season_end_year(season_tag)

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    for s in SEASONS:
        print(f"[INFO] loading: {s}")
        parts.append(load_one_season(s))

    master = pd.concat(parts, ignore_index=True)

    meta = ["season", "player_name", "position", "team_abbr"]
    cols = meta + [c for c in master.columns if c not in meta]
    master = master[cols]

    out_path = PROCESSED_DIR / "players_stats_master.csv"
    master.to_csv(out_path, index=False)
    print(f"[OK] wrote: {out_path} | rows={len(master)} cols={master.shape[1]}")


if __name__ == "__main__":
    main()