import json
r=json.load(open('reports/G07_full_racket_v4/cem27_plane_terminal_1925/REPORT.json'));assert r['completed_generations']==1 and r['counts']['attack_outgoing']==0;print('CEM27 terminal verified')
