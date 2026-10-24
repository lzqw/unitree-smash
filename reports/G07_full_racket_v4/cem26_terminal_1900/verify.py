import json
r=json.load(open('reports/G07_full_racket_v4/cem26_terminal_1900/REPORT.json'));assert r['completed_generations']==4 and r['counts']['complete_smash']==0 and r['counts']['numerical_failures']==0;assert r['best_native_generation3_candidate43']['norm_mps']>=8 and not r['best_native_generation3_candidate43']['complete'];print('CEM26 terminal verified')
