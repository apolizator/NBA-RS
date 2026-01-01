"""
build_team_profiles_from_file.py

Amaç:
- Kullanıcının verdiği hazır takım istatistik CSV'sini
  (data/raw/team_stats_raw.csv) okuyup,
  sezon + takım bazlı istatistik profiline ve z-score'lara çevirmek.

Girdi:
- data/raw/team_stats_raw.csv  (tek sezon da olabilir, örn. 2024-2025)

Çıktı:
- data/processed/team_stats_profiles.csv

Notlar:
- Eğer sezon kolonu yoksa, tüm satırlara season = 2025 atanır.
- Z-score hesabında 'season', 'Rk' ve 'G' kolonları kullanılmaz.
- Tüm float kolonlar (özellikle *_z) CSV'ye yazılmadan önce 4 ondalığa yuvarlanır.
- Takım adlarındaki '*' gibi işaretler temizlenir (örn. "Los Angeles Lakers*" -> "Los Angeles Lakers").
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from config import PROCESSED_DIR, TEAM_STATS_RAW_PATH


TEAM_PROFILES_PATH = PROCESSED_DIR / "team_stats_profiles.csv"


def load_raw_team_stats() -> Tuple[pd.DataFrame, List[str]]:
    """Ham takım istatistik dosyasını yükler, kolonları normalize eder ve feature listesini döner."""
    if not TEAM_STATS_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Takım istatistik dosyası bulunamadı: {TEAM_STATS_RAW_PATH}"
        )

    print(f"[INFO] Takım istatistikleri yükleniyor: {TEAM_STATS_RAW_PATH}")
    df = pd.read_csv(TEAM_STATS_RAW_PATH)

    # --- sezon & takım kolonlarını otomatik bul ---
    SEASON_COL_CANDIDATES = ["season", "Season", "SEASON", "year", "Year", "YEAR"]
    TEAM_COL_CANDIDATES = ["team", "Team", "TEAM", "Tm", "TM"]

    def find_column(candidates, logical_name: str) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"'{logical_name}' için uygun kolon bulunamadı. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    # Takım kolonu ZORUNLU
    team_col = find_column(TEAM_COL_CANDIDATES, "team")
    df = df.rename(columns={team_col: "team"})

    # Takım adlarındaki '*' gibi işaretleri temizle
    df["team"] = df["team"].astype(str).str.replace("*", "", regex=False).str.strip()

    # Sezon kolonu opsiyonel: yoksa hepsine 2025 ver
    try:
        season_col = find_column(SEASON_COL_CANDIDATES, "season")
        df = df.rename(columns={season_col: "season"})
    except ValueError:
        print("[INFO] Sezon kolonu bulunamadı, tüm satırlara season = 2025 atanıyor...")
        df["season"] = 2025

    # Numeric kolonlar
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Z-score hesabında KULLANMAYACAĞIMIZ kolonlar:
    # - season: zaten grouping değişkeni
    # - Rk: sıralama, anlamsız
    # - G: tek sezonda genelde tüm takımlar için aynı (std=0, z-score = 0)
    EXCLUDE_FOR_FEATURES = {"season", "Rk", "G"}

    feature_cols = [c for c in num_cols if c not in EXCLUDE_FOR_FEATURES]

    if not feature_cols:
        raise ValueError(
            "Takım istatistik dosyasında kullanılabilir numeric istatistik kolonu bulunamadı."
        )

    print(f"[INFO] Kullanılacak feature kolonları (z-score hesaplanacak): {feature_cols}")

    # Aynı sezon+takım için birden fazla satır varsa ortalama al
    group_cols = ["season", "team"]
    df_grouped = (
        df.groupby(group_cols)[feature_cols]
        .mean()
        .reset_index()
    )

    return df_grouped, feature_cols


def add_zscores_per_season(team_stats: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Her sezon için ayrı ayrı:
    - Her istatistik kolonu için lig ortalamasına göre z-score hesaplar.
      z = (x - mean) / std

    Tek sezonluk dosya olsa bile aynı mantık geçerli (sadece 1 sezon loop eder).
    """
    frames = []

    for season in sorted(team_stats["season"].unique()):
        ts = team_stats[team_stats["season"] == season].copy()

        for col in feature_cols:
            mean = ts[col].mean()
            std = ts[col].std(ddof=0)

            if std == 0 or pd.isna(std):
                ts[f"{col}_z"] = 0.0
            else:
                ts[f"{col}_z"] = (ts[col] - mean) / std

        frames.append(ts)

    result = pd.concat(frames, ignore_index=True)
    return result


def build_team_profiles_from_file() -> Path:
    """Ham dosyadan takım profilleri + z-score'lar üretir, 4 ondalığa yuvarlayıp CSV'ye kaydeder."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    team_stats, feature_cols = load_raw_team_stats()
    team_profiles = add_zscores_per_season(team_stats, feature_cols)

    # --- Ondalıkları 4 haneye yuvarla ---
    float_cols = team_profiles.select_dtypes(include=["float64", "float32"]).columns
    team_profiles[float_cols] = team_profiles[float_cols].round(4)

    team_profiles.to_csv(TEAM_PROFILES_PATH, index=False)
    print(f"[INFO] Takım profilleri kaydedildi: {TEAM_PROFILES_PATH}")

    return TEAM_PROFILES_PATH


def main() -> None:
    build_team_profiles_from_file()


if __name__ == "__main__":
    main()