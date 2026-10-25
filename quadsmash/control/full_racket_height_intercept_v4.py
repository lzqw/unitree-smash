"""Select an own-side interception plane from a current public ball forecast.

The target height is a policy choice, not a relaxed task threshold. Every
candidate uses the locked shuttle model. This module commands no robot or
shuttle state and provides no feasibility or success guarantee.
"""
import numpy as np

from .initial_intercept import predict_initial_intercept


def validate_height_intercept(config):
    if config is None:
        return None
    height=float(config['target_height_m'])
    bounds=np.asarray(config.get('plane_bounds_m',[.8,1.45]),dtype=float)
    count=config.get('plane_count',33)
    fraction=float(config.get('adaptation_fraction',1.))
    if not np.isfinite(height) or not 1.65<=height<=1.95:
        raise ValueError('Require a finite policy target height in[1.65,1.95]m')
    if (bounds.shape!=(2,) or not np.isfinite(bounds).all() or
            not .8<=bounds[0]<bounds[1]<=1.45):
        raise ValueError('Planes must stay inside the existing[.8,1.45]m policy search range')
    if type(count) is not int or not 3<=count<=65:
        raise ValueError('Require3..65candidate planes')
    if not np.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('Require an adaptation fraction in[0,1]')
    return dict(target_height_m=height,plane_bounds_m=bounds.tolist(),
                plane_count=count,adaptation_fraction=fraction)


def predict_height_intercept(public,command_position,command_normal,reference_x,
                             physics,config,dt=.001,max_seconds=1.5):
    """Predict before choosing; never read a case ID or future simulator label.

    The existing interception plane remains the fallback. Invalid forecasts
    neither discard a case nor revive its rally. Zero adaptation takes the
    original predictor path exactly, including its validity flags.
    """
    config=validate_height_intercept(config)
    public=np.asarray(public)
    if public.ndim!=2 or public.shape[1]!=794:
        raise ValueError('Require N794 current public observations')
    reference=np.broadcast_to(reference_x,(len(public),)).copy()
    if not np.isfinite(reference).all() or (reference<.8).any() or (reference>1.45).any():
        raise ValueError('Reference planes outside existing policy bounds')
    if config is None or config['adaptation_fraction']==0.:
        return predict_initial_intercept(public,command_position,command_normal,reference,
                                         physics,dt=dt,max_seconds=max_seconds)
    if ((reference<config['plane_bounds_m'][0]).any() or
            (reference>config['plane_bounds_m'][1]).any()):
        raise ValueError('Reference planes must lie inside the configured candidate range')
    count=config['plane_count']
    grid=np.linspace(*config['plane_bounds_m'],count)
    # All inputs are the same current observation; only the proposed plane
    # changes. Runtime does not open recorded episodes or formal case files.
    plans=predict_initial_intercept(np.repeat(public,count,axis=0),command_position,
                                    command_normal,np.tile(grid,len(public)),physics,
                                    dt=dt,max_seconds=max_seconds)
    points=plans['position_m'].reshape(len(public),count,3)
    arrivals=plans['arrival_time_s'].reshape(len(public),count)
    valid=plans['valid'].reshape(len(public),count)
    absolute=public[:,775:776]+arrivals
    valid &= np.isfinite(points).all(-1)&np.isfinite(absolute)&(absolute>=2.3)&(absolute<=3.3)
    error=abs(points[:,:,2]-config['target_height_m'])
    error+=1e-8*abs(grid[None]-reference[:,None])
    score=np.where(valid,error,np.inf)
    best=score.argmin(1)
    available=valid.any(1)
    selected=np.where(available,grid[best],reference)
    chosen=reference+config['adaptation_fraction']*(selected-reference)
    final=predict_initial_intercept(public,command_position,command_normal,chosen,
                                    physics,dt=dt,max_seconds=max_seconds)
    final.update(selected_intercept_plane_x_m=chosen,
                 height_adaptation_available=available,
                 requested_height_m=np.full(len(public),config['target_height_m']))
    return final
