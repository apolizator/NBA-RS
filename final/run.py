"""Run the project end-to-end.

This is an orchestration wrapper. It does NOT change any scoring logic.
It simply runs the existing scripts in the correct order and fails fast
if any step fails.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    py = sys.executable
    run([py, "src/load_player_stats.py"])
    run([py, "src/build_weighted_player_profiles.py"])
    run([py, "src/build_team_profiles_from_file.py"])
    run([py, "src/calculate_team_needs.py"])
    run([py, "src/recommender.py", "--mode", "draft"])
    run([py, "src/compute_team_position_needs.py"])
    run([py, "src/find_cross_team_swaps.py"])
    run([py, "src/report_filtered_recs.py"])
    run([py, "f.py"])
    print("\n✅ completed. outputs: data/processed/")


if __name__ == "__main__":
    main()
