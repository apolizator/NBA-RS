"""
find_cross_team_swaps.py

Amaç:
- team_player_recs_summer_2024_fixed.csv dosyasındaki önerilerden,
  birbirlerine karşılıklı oyuncu önerilen takım çiftlerini (A ↔ B)
  ve bu takımlar arasındaki olası 1'e 1 takas çekirdeklerini çıkarmak.

Mantık:
- last_team = oyuncunun önceki takımı (2023-24)
- team_abbr = modelin önerdiği yeni takım

A -> B akışı:
    last_team = A, team_abbr = B olan öneriler
B -> A akışı:
    last_team = B, team_abbr = A olan öneriler

Eğer hem A -> B hem B -> A varsa:
    => Çapraz eşleşen takım çifti (swap potansiyeli)
    => Bu iki yöndeki oyuncular arasında tüm olası 1'e 1 kombinasyonlar
       CSV'ye yazılır.

Filtreleme (Pozisyon Bazlı):
- Sadece şu pozisyon eşleşmeleri kabul edilir:
  PG ↔ C, SG ↔ PF, SF ↔ SF

Çıktılar:
- Konsol:
    Her takım çifti için özet liste
- CSV:
    data/processed/cross_team_swap_candidates.csv
    Kolonlar:
      team_a, team_b,
      from_a_player, from_a_position, from_a_fit_score,
      from_b_player, from_b_position, from_b_fit_score
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from config import PROCESSED_DIR

RECS_PATH = PROCESSED_DIR / "team_player_recs_summer_2024_fixed.csv"
OUT_PATH = PROCESSED_DIR / "cross_team_swap_candidates.csv"


def main() -> None:
    if not RECS_PATH.exists():
        raise FileNotFoundError(f"Öneri dosyası bulunamadı: {RECS_PATH}")

    df = pd.read_csv(RECS_PATH)

    required = {"player_name", "team_abbr", "last_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"team_player_recs_summer_2024_fixed.csv içinde şu kolon(lar) eksik: {missing}"
        )

    df = df.copy()

    # İhtiyaç duyabileceğimiz kolonların tiplerini düzelt
    df["last_team"] = df["last_team"].astype(str)
    df["team_abbr"] = df["team_abbr"].astype(str)

    # Pozisyon ve fit_score varsa al, yoksa dummy doldur
    if "target_position" not in df.columns:
        df["target_position"] = ""
    if "fit_score" not in df.columns:
        df["fit_score"] = float("nan")
    else:
        df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce")

    # Sadece gerçekten takım değiştirme önerilerine bakalım:
    # last_team dolu ve last_team != team_abbr
    df_moves = df[
        df["last_team"].notna()
        & (df["last_team"] != "")
        & (df["last_team"] != df["team_abbr"])
    ].copy()

    if df_moves.empty:
        print("Hiç 'takım değiştirme' önerisi bulunamadı (last_team != team_abbr satırı yok).")
        return

    # Akışlar: (from_team, to_team) -> satır index listesi
    flows: Dict[Tuple[str, str], List[int]] = {}
    for idx, row in df_moves.iterrows():
        from_team = row["last_team"]
        to_team = row["team_abbr"]
        key = (from_team, to_team)
        flows.setdefault(key, []).append(idx)

    seen_pairs = set()
    swap_rows: List[dict] = []

    print("=== Çapraz Takım Eşleşmeleri ve Olası Takas Çekirdekleri (A ↔ B) ===\n")

    for (a, b), idx_list_ab in flows.items():
        # Ters yön (B -> A) yoksa çapraz yok
        if (b, a) not in flows:
            continue

        # Aynı takım çiftini iki kere yazmamak için
        pair_key = tuple(sorted((a, b)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        idx_list_ba = flows[(b, a)]

        players_ab = df_moves.loc[idx_list_ab, [
            "player_name", "target_position", "fit_score"
        ]].copy()
        players_ba = df_moves.loc[idx_list_ba, [
            "player_name", "target_position", "fit_score"
        ]].copy()

        print(f"{a} ↔ {b}")
        print(f"  {a} → {b}:")
        for _, r in players_ab.iterrows():
            name = r["player_name"]
            pos = r.get("target_position", "")
            score = r.get("fit_score", float("nan"))
            if pd.notna(score):
                print(f"    - {name} ({pos}, fit={score:.3f})")
            else:
                print(f"    - {name} ({pos})")

        print(f"  {b} → {a}:")
        for _, r in players_ba.iterrows():
            name = r["player_name"]
            pos = r.get("target_position", "")
            score = r.get("fit_score", float("nan"))
            if pd.notna(score):
                print(f"    - {name} ({pos}, fit={score:.3f})")
            else:
                print(f"    - {name} ({pos})")

        print()

        # 1'e 1 tüm kombinasyonları CSV için üret
        for _, r_ab in players_ab.iterrows():
            for _, r_ba in players_ba.iterrows():
                # Pozisyon filtresi: PG-C, SG-PF, SF-SF
                pos_a = str(r_ab.get("target_position", "")).strip().upper()
                pos_b = str(r_ba.get("target_position", "")).strip().upper()
                pair = tuple(sorted((pos_a, pos_b)))

                if pair not in {("C", "PG"), ("PF", "SG"), ("SF", "SF")}:
                    continue

                # Toplam fit_score hesapla
                val_a = r_ab.get("fit_score", 0)
                val_b = r_ba.get("fit_score", 0)
                score_a = float(val_a) if pd.notna(val_a) else 0.0
                score_b = float(val_b) if pd.notna(val_b) else 0.0
                total_score = score_a + score_b

                swap_rows.append(
                    {
                        "team_a": a,
                        "team_b": b,
                        "from_a_player": r_ab["player_name"],
                        "from_a_position": r_ab.get("target_position", ""),
                        "from_a_fit_score": (
                            round(float(r_ab["fit_score"]), 4)
                            if pd.notna(r_ab["fit_score"])
                            else ""
                        ),
                        "from_b_player": r_ba["player_name"],
                        "from_b_position": r_ba.get("target_position", ""),
                        "from_b_fit_score": (
                            round(float(r_ba["fit_score"]), 4)
                            if pd.notna(r_ba["fit_score"])
                            else ""
                        ),
                        "total_fit_score": round(total_score, 4),
                    }
                )

    if not swap_rows:
        print("Hiç çapraz takım eşleşmesi (A → B ve B → A) bulunamadı.")
        return

    out_df = pd.DataFrame(swap_rows)
    out_df = out_df.sort_values(by="total_fit_score", ascending=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"✅ Olası çapraz takas kombinasyonları CSV'ye kaydedildi: {OUT_PATH}")


if __name__ == "__main__":
    main()