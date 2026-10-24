import json
r=json.load(open('reports/G07_full_racket_v4/relative_prepose_two_gen_terminal_1748/REPORT.json'));assert r['completed_generations']==2 and r['completed_counts']['attack_outgoing']==0 and r['completed_counts']['complete_smash']==0 and r['interrupted_ticks']==27;print('CEM25b terminal verified')
