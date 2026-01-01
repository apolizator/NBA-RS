import pandas as pd
from config import PROCESSED_DIR
from constants import TEAM_NAME_TO_ABBR

TEAM_PROFILES_PATH = PROCESSED_DIR / "team_stats_profiles.csv"
TEAM_NEEDS_PATH = PROCESSED_DIR / "team_needs.csv"

TARGET_TEAM_SEASON = 2025

def clean_team_name(name: str) -> str:
    return name.replace("*", "").strip() if isinstance(name, str) else ""

def normalize_series(s: pd.Series, min_val: float = 0.2, max_val: float = 1.0) -> pd.Series:
    """Seriyi min_val ile max_val arasına sıkıştırır. Asla 0 dönmez."""
    s_min, s_max = s.min(), s.max()
    if s_max == s_min:
        return pd.Series(max_val, index=s.index)
    # 0-1 arası normalizasyon
    norm = (s - s_min) / (s_max - s_min)
    # İstenen aralığa çekme (örn: 0.2 - 1.0)
    return min_val + norm * (max_val - min_val)

def compute_team_scoring_defense_needs(df_team: pd.DataFrame) -> pd.DataFrame:
    df = df_team[df_team["season"] == TARGET_TEAM_SEASON].copy()
    
    # Calculate Raw Needs
    for idx, row in df.iterrows():
        def get_z(c): return float(row[c]) if c in row.index else 0.0
        
        # Z-score ne kadar düşükse ihtiyaç o kadar yüksektir.
        # Bu yüzden -1 ile çarpıyoruz. (Düşük performans -> Yüksek Ham İhtiyaç)
        df.loc[idx, "Need_scoring_raw"] = -1 * (get_z("PTS_z") + get_z("FG%_z") + get_z("3P%_z"))
        df.loc[idx, "Need_defense_raw"] = -1 * (get_z("DRB_z") + get_z("STL_z") + get_z("BLK_z"))
    
    # Normalize to [0.2, 1.0] -> ASLA 0 OLMAZ
    df["Need_scoring"] = normalize_series(df["Need_scoring_raw"], 0.2, 1.0)
    df["Need_defense"] = normalize_series(df["Need_defense_raw"], 0.2, 1.0)
    
    # Clean and Map Names
    df["team_clean"] = df["team"].apply(clean_team_name)
    df["team_abbr"] = df["team_clean"].map(TEAM_NAME_TO_ABBR)
    
    # Return only relevant columns
    return df[["team", "team_abbr", "Need_scoring", "Need_defense"]]

def main():
    if not TEAM_PROFILES_PATH.exists():
        raise FileNotFoundError(f"{TEAM_PROFILES_PATH} bulunamadı.")
        
    print(f"[INFO] Takım profilleri okunuyor: {TEAM_PROFILES_PATH}")
    df_team = pd.read_csv(TEAM_PROFILES_PATH)
    
    print("[INFO] Takım ihtiyaç vektörleri hesaplanıyor...")
    df_needs = compute_team_scoring_defense_needs(df_team)
    
    df_needs.to_csv(TEAM_NEEDS_PATH, index=False)
    print(f"✅ Takım ihtiyaç vektörü kaydedildi: {TEAM_NEEDS_PATH}")

if __name__ == "__main__":
    main()