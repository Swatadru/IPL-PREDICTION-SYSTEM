import csv, json
from collections import defaultdict

matches = list(csv.DictReader(open('static/ipl_prediction/ipl_data/historical_matches.csv')))
teams_raw = list(csv.DictReader(open('static/ipl_prediction/ipl_data/teams.csv')))
teams = {r['team_id']: r for r in teams_raw}
stadiums_raw = list(csv.DictReader(open('static/ipl_prediction/ipl_data/stadiums.csv')))

# 1. Head-to-head records (by team name pairs)
h2h = defaultdict(lambda: defaultdict(int))
for m in matches:
    t1, t2, w = m['team1_id'], m['team2_id'], m['winner']
    n1, n2 = teams[t1]['name'], teams[t2]['name']
    key = tuple(sorted([n1, n2]))
    h2h[key]['total'] += 1
    if w == t1:
        h2h[key][n1] += 1
    elif w == t2:
        h2h[key][n2] += 1

# 2. Team win rate overall (from actual match data)
team_wins = defaultdict(lambda: {'wins': 0, 'total': 0})
for m in matches:
    t1, t2, w = m['team1_id'], m['team2_id'], m['winner']
    n1, n2 = teams[t1]['name'], teams[t2]['name']
    team_wins[n1]['total'] += 1
    team_wins[n2]['total'] += 1
    if w == t1:
        team_wins[n1]['wins'] += 1
    elif w == t2:
        team_wins[n2]['wins'] += 1

# 3. Toss advantage
toss_total = len(matches)
toss_win_match = sum(1 for m in matches if m['toss_winner'] == m['winner'])
print(f"TOSS_ADV: {toss_win_match}/{toss_total} = {toss_win_match/toss_total*100:.2f}%")

# 4. Bat first vs Field first
bat_total = sum(1 for m in matches if m['toss_decision'] == 'Bat')
bat_wins = sum(1 for m in matches if m['toss_decision'] == 'Bat' and m['toss_winner'] == m['winner'])
field_total = toss_total - bat_total
field_wins = toss_win_match - bat_wins
print(f"BAT_FIRST_WIN: {bat_wins}/{bat_total} = {bat_wins/max(bat_total,1)*100:.2f}%")
print(f"FIELD_FIRST_WIN: {field_wins}/{field_total} = {field_wins/max(field_total,1)*100:.2f}%")

# 5. Team win rates
print("\n=== TEAM WIN RATES (from 7200 matches) ===")
for name in sorted(team_wins.keys()):
    t = team_wins[name]
    pct = t['wins'] / max(t['total'], 1) * 100
    print(f"  {name}: {t['wins']}/{t['total']} = {pct:.2f}%")

# 6. Team avg batting scores
team_bat = defaultdict(lambda: {'total': 0, 'count': 0})
for m in matches:
    n1 = teams[m['team1_id']]['name']
    team_bat[n1]['total'] += int(m['first_innings_score'])
    team_bat[n1]['count'] += 1
    n2 = teams[m['team2_id']]['name']
    team_bat[n2]['total'] += int(m['second_innings_score'])
    team_bat[n2]['count'] += 1

print("\n=== TEAM AVG SCORES ===")
for name in sorted(team_bat.keys()):
    t = team_bat[name]
    avg = t['total'] / max(t['count'], 1)
    print(f"  {name}: {avg:.1f}")

# 7. Team win rate at each stadium
team_venue_wins = defaultdict(lambda: {'wins': 0, 'total': 0})
stadiums_map = {s['stadium_id']: s['name'] for s in stadiums_raw}
for m in matches:
    n1, n2 = teams[m['team1_id']]['name'], teams[m['team2_id']]['name']
    sname = stadiums_map.get(m['stadium_id'], 'Unknown')
    team_venue_wins[(n1, sname)]['total'] += 1
    team_venue_wins[(n2, sname)]['total'] += 1
    winner_name = teams.get(m['winner'], {}).get('name', '')
    if winner_name == n1:
        team_venue_wins[(n1, sname)]['wins'] += 1
    elif winner_name == n2:
        team_venue_wins[(n2, sname)]['wins'] += 1

# 8. Build head-to-head JSON
print("\n=== HEAD TO HEAD (sample) ===")
h2h_data = {}
for key, val in h2h.items():
    pair_key = f"{key[0]} vs {key[1]}"
    h2h_data[pair_key] = dict(val)
    if val['total'] > 50:
        t1_name, t2_name = key
        t1w = val.get(t1_name, 0)
        t2w = val.get(t2_name, 0)
        print(f"  {t1_name} vs {t2_name}: {t1w}-{t2w} (total {val['total']})")

# 9. Export compact JSON data for embedding
output = {
    'team_win_rates': {},
    'team_avg_scores': {},
    'h2h': {},
    'team_venue': {},
    'toss_advantage': round(toss_win_match / toss_total, 4),
    'bat_first_win_rate': round(bat_wins / max(bat_total, 1), 4),
    'field_first_win_rate': round(field_wins / max(field_total, 1), 4)
}

for name in team_wins:
    t = team_wins[name]
    output['team_win_rates'][name] = round(t['wins'] / max(t['total'], 1), 4)

for name in team_bat:
    t = team_bat[name]
    output['team_avg_scores'][name] = round(t['total'] / max(t['count'], 1), 1)

for key, val in h2h.items():
    t1_name, t2_name = key
    pair = f"{t1_name}|{t2_name}"
    output['h2h'][pair] = {
        'total': val['total'],
        t1_name: val.get(t1_name, 0),
        t2_name: val.get(t2_name, 0)
    }

for (tname, sname), val in team_venue_wins.items():
    k = f"{tname}|{sname}"
    output['team_venue'][k] = round(val['wins'] / max(val['total'], 1), 4)

with open('ml_stats.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n=== EXPORTED ml_stats.json ===")
print(f"H2H pairs: {len(output['h2h'])}")
print(f"Team-venue combos: {len(output['team_venue'])}")
