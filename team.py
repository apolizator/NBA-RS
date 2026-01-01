import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster

def fetch_nba_rosters():
    # 1. Get list of all NBA teams (for ID and Abbreviations)
    nba_teams = teams.get_teams()
    all_players_list = []

    print("🚀 Fetching NBA Current Roster Data...")
    print("------------------------------------------")

    for team in nba_teams:
        team_id = team['id']
        team_abbr = team['abbreviation']
        
        try:
            # 2. Fetch current roster for each team from NBA API
            roster_data = commonteamroster.CommonTeamRoster(team_id=team_id)
            df_roster = roster_data.get_data_frames()[0]
            
            # 3. Extract only Player Name and Team Abbreviation
            for _, row in df_roster.iterrows():
                all_players_list.append({
                    'player_name': row['PLAYER'],
                    'team_abbr': team_abbr
                })
            
            print(f"✅ {team['full_name']} ({team_abbr}) roster fetched.")
            
            # Short pause to avoid getting banned from NBA servers (Rate Limit)
            time.sleep(0.6) 
            
        except Exception as e:
            print(f"❌ Error fetching {team_abbr}: {e}")

    # 4. Convert data to DataFrame and save as CSV
    final_df = pd.DataFrame(all_players_list)
    
    # Clean whitespace in names (to prevent matching errors)
    final_df['player_name'] = final_df['player_name'].str.strip()
    
    # Sort alphabetically (Optional, looks neater)
    final_df = final_df.sort_values(by=['team_abbr', 'player_name'])

    # Save as CSV
    final_df.to_csv("nba_current_roster_2025.csv", index=False, encoding='utf-8')
    
    print("------------------------------------------")
    print(f"🏁 Process completed successfully!")
    print(f"📂 Total {len(final_df)} players saved to 'nba_current_roster_2025.csv'.")

if __name__ == "__main__":
    fetch_nba_rosters()