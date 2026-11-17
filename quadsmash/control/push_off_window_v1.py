"""Bounded causal support window; no reactivation after flight or handoff."""
import numpy as np


class PushOffWindowV1:
    def __init__(self,contact_index):
        if type(contact_index) is not int or not 2<=contact_index<=96:
            raise ValueError('Explicit bounded contact node required')
        self.deadline=contact_index-1
        self.reset()

    def reset(self):
        self.last_node=-1;self.stopped=None

    def step(self,node,contacts,*,recovery_owned=False):
        contacts=np.asarray(contacts)
        if type(node) is not int or node!=self.last_node+1:
            raise ValueError('Sequential current-state updates required')
        if contacts.shape!=(4,) or contacts.dtype!=np.bool_ or type(recovery_owned) is not bool:
            raise ValueError('Explicit measured/predicted Boolean contacts and ownership required')
        self.last_node=node
        if recovery_owned:self.stopped='RECOVERY_OWNED'
        elif self.stopped is None:
            if not contacts.any():self.stopped='FLIGHT_OBSERVED'
            elif node>=self.deadline:self.stopped='CONTACT_DEADLINE'
        return dict(enabled=self.stopped is None,phase='PUSH_OFF' if self.stopped is None else self.stopped,
            node=node,deadline_node=self.deadline,contacts=contacts.copy())
