from typing import Dict, Set

UNTOUCHABLE_PLAYERS: Set[str] = {
    "Nikola Jokić", "Victor Wembanyama", "Giannis Antetokounmpo", "Stephen Curry",
    "LeBron James", "Joel Embiid", "Jayson Tatum", "Shai Gilgeous-Alexander",
    "James Harden", "Kyrie Irving", "Kawhi Leonard", "Devin Booker",
    "Donovan Mitchell", "Trae Young", "Ja Morant", "Anthony Edwards",
    "Tyrese Haliburton", "Fred VanVleet", "Nikola Vučević", "Jaylen Brown",
    "Bam Adebayo", "Jamal Murray", "LaMelo Ball", "Chet Holmgren",
    "Tyler Herro", "Jaren Jackson Jr."
}

TEAM_NAME_TO_ABBR: Dict[str, str] = {
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

ABBR_TO_TEAM_NAME: Dict[str, str] = {abbr: name for name, abbr in TEAM_NAME_TO_ABBR.items()}