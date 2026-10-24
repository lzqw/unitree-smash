import json
r=json.load(open('reports/G07_full_racket_v4/cem28_terminal_2030/REPORT.json'));assert r['completed_generations']==4 and r['counts']['complete_smash']>=2 and r['counts']['numerical_failures']==0;print('CEM28 terminal verified with',r['counts']['complete_smash'],'complete native TRAIN episodes')
