"""
compute_team_position_needs.py

Amaç:
- team_player_recs_summer_2024_fixed.csv çıktısını kullanarak
  "hangi takımın hangi pozisyonda göreli olarak daha fazla ihtiyacı var?"
  sorusuna basit bir cevap üretmek.

Heuristik:
- Her takım için:
    team_mean = bu takıma önerilen TÜM oyuncuların ortalama fit_score'u
- Her (takım, pozisyon) için:
    pos_mean = bu takım + pozisyon için önerilen oyuncuların ortalama fit_score'u

- need_score = pos_mean - team_mean

- need_score > 0 ise:
    -> Bu pozisyonda fit_score ortalaması takım geneline göre daha yüksek,
       model bu pozisyon için daha "ihtiyaç odaklı" öneri üretmiş.

Ek düzeltme (star anchor filtresi):
- Bazı takımlarda belirli pozisyonlarda zaten süper yıldız var
  (ör: GSW–PG = Curry, DEN–C = Jokic).
- Bu pozisyonları "ANCHOR" kabul edip, ihtiyaç listesine koymuyoruz.
"""

from __future__ import annotations

import pandas as pd

from config import PROCESSED_DIR

RECS_PATH = PROCESSED_DIR / "team_player_recs_summer_2024_fixed.csv"
OUT_PATH = PROCESSED_DIR / "team_position_needs.csv"

def main() -> None:
    if not RECS_PATH.exists():
        raise FileNotFoundError(f"Öneri dosyası bulunamadı: {RECS_PATH}")

    df = pd.read_csv(RECS_PATH)

    required = {"team_abbr", "target_position", "fit_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"team_player_recs_summer_2024_fixed.csv içinde şu kolonlar eksik: {missing}"
        )

    df = df.copy()
    df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce")
    df = df.dropna(subset=["fit_score"])

    # Takım genel ortalama fit_score
    team_mean = (
        df.groupby("team_abbr")["fit_score"]
        .mean()
        .rename("team_mean")
        .reset_index()
    )

    # Takım + pozisyon ortalama fit_score
    pos_mean = (
        df.groupby(["team_abbr", "target_position"])["fit_score"]
        .mean()
        .rename("pos_mean")
        .reset_index()
    )

    # Birleştir
    merged = pos_mean.merge(team_mean, on="team_abbr", how="left")

    # Göreli ihtiyaç skoru
    merged["need_score"] = merged["pos_mean"] - merged["team_mean"]

    # Sıralama: takım + need_score DESC
    merged = merged.sort_values(
        ["team_abbr", "need_score"],
        ascending=[True, False]
    )

    # Konsola okunabilir özet
    print("=== TAKIM + POZİSYON BAZLI İHTİYAÇ SIRALAMASI (TÜM POZİSYONLAR) ===\n")

    for team, sub in merged.groupby("team_abbr"):
        # Tüm pozisyonları al, filtreleme yapma
        needs = sub.copy()

        lines = []
        for _, row in needs.iterrows():
            pos = row["target_position"]
            need_score = row["need_score"]
            pos_mean = row["pos_mean"]
            team_mean_val = row["team_mean"]
            lines.append(
                f"{pos} (need_score={need_score:.3f}, pos_mean={pos_mean:.3f}, team_mean={team_mean_val:.3f})"
            )

        joined = " | ".join(lines)
        print(f"{team}: ihtiyaç pozisyonları -> {joined}")

    # CSV olarak kaydet
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Pozisyon bazlı ihtiyaç tablosu kaydedildi: {OUT_PATH}")


if __name__ == "__main__":
    main()