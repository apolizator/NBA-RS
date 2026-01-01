from pathlib import Path

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Veri klasörleri
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Çalışacağımız sezonlar (örnek)
# Buradaki sezondan kastım: "2020" = 2019-20 sezonu gibi düşünebilirsin.
SEASONS = [2020, 2021, 2022, 2023, 2024]

# Her sezon için hangi CSV dosyasını kullanacağımızı burada tanımlıyoruz.
# >>> BURAYI KENDİ DOSYA İSİMLERİNE GÖRE DÜZENLE <<<
SEASON_STATS_FILES = {
    2020: RAW_DIR / "player_stats_2019_20.csv",
    2021: RAW_DIR / "player_stats_2020_21.csv",
    2022: RAW_DIR / "player_stats_2021_22.csv",
    2023: RAW_DIR / "player_stats_2022_23.csv",
    2024: RAW_DIR / "player_stats_2023_24.csv",
}

# Aktif NBA takımları (kadro ve filtreleme için)
NBA_TEAMS = [
    "ATL", "BOS", "BRK", "CHI", "CHO", "CLE", "DAL", "DEN", "DET",
    "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
    "NOP", "NYK", "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS",
    "TOR", "UTA", "WAS",
]
TEAM_STATS_RAW_PATH = RAW_DIR / "team_stats_raw.csv"