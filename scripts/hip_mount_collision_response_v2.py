"""Retry a failed coupled contact iteration without relaxing its equations.

The ordinary V1 result remains unchanged when it converges. Only iteration
exhaustion of a contact fixed-point solve permits a bounded retry. Geometry,
momentum checks, friction, motor limits and numerical tolerances are retained.
This is a model candidate, not a native physical parameter modification.
"""
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1


class HipMountCollisionResponseV2(HipMountCollisionResponseV1):
    def __init__(self, model, *, retry_iterations=4000, penetration_decay_s=.01):
        super().__init__(model,penetration_decay_s=penetration_decay_s)
        if type(retry_iterations) is not int or not 400<retry_iterations<=4000:
            raise ValueError('Explicit finite extra solver budget required')
        self.retry_iterations=retry_iterations
        self.retries=[]

    def step(self,*args,**kwargs):
        try:
            return super().step(*args,**kwargs)
        except ValueError as error:
            allowed=('Coupled contact/joint interval law did not converge:',
                'Coupled contact/speed law did not converge:',
                'Coulomb contact law did not converge:')
            if not str(error).startswith(allowed) or kwargs.get('max_iterations',400)>=self.retry_iterations:
                raise
            event=dict(original_error=str(error),retry_iterations=self.retry_iterations,converged=False)
            self.retries.append(event)
            result=super().step(*args,**dict(kwargs,max_iterations=self.retry_iterations))
            event.update(converged=True,momentum_residual=result['momentum_residual'],
                contact_iterations=result['contact']['iterations'],contact_residual=result['contact']['residual'])
            return result
