"""Compare repeated native load events and contact topology across four signed probe executions."""

def signatures(trace):
    t=trace['tensors']
    nr=t['racket_force_w'][1:].norm(dim=-1)>.001
    fr=t['racket_friction_force_w'][1:].norm(dim=-1)>.001
    grd=t['ground_force_w'][1:].norm(dim=-1)>.001
    net=t['net_force_matrix_w'][1:].reshape(*nr.shape,-1,3).norm(dim=-1).amax(-1)>.001
    result=[]
    for lane in range(nr.shape[1]):
        frames=(nr[:,lane]|fr[:,lane]).nonzero().flatten().tolist()
        points=[]
        for i in frames:
            m=trace['manifold_frames'][i+1];f=trace['racket_friction_frames'][i+1]
            points.append((int(((m['lane']==lane)&(m['filter_index']==0)&(m['normal_force_n'].abs()>.001)).sum()),
                           int(((f['lane']==lane)&(f['force_w_n'].norm(dim=-1)>.001)).sum())))
        result.append((bool(grd[:,lane].any()),bool(net[:,lane].any()),bool(frames),tuple(frames),tuple(points)))
    return result


def classify(rows):
    if rows[0]!=rows[1]:return 'nominal_mismatch'
    if any(r[:3]!=rows[0][:3] for r in rows[2:]):return 'object_switch'
    if any(rows[0][:2]):return 'nonracket_contact'
    if not rows[0][2]:return 'no_contact'
    if any(r[3]!=rows[0][3] for r in rows[2:]):return 'toi_switch'
    if any(r[4]!=rows[0][4] for r in rows[2:]):return 'manifold_switch'
    return 'racket_stable'


