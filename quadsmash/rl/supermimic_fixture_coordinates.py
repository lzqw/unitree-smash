"""Use each native environment's own court for fixture ball-rollout diagnostics."""
import torch


def fixture_court_origin(packet, initial_position, design=None):
    layout = packet.get('fixture_layout')
    registered = (design or {}).get('fixture_layout')
    if layout is None:
        if registered is not None:
            raise ValueError('Registered layout is missing from the native fixture')
        # Historical fixtures have no saved origins. Preserve historical reports.
        return torch.zeros_like(initial_position)
    origin = initial_position.new_tensor(layout['origins_w'])
    if origin.shape != initial_position.shape or not origin.isfinite().all():
        raise ValueError('Invalid native court origins')
    if layout['environment_spacing_m'] == 0 and torch.count_nonzero(origin):
        raise ValueError('Nonzero measured origin in compact fixture')
    if registered is not None and (layout['mode'] != registered or
            layout['environment_spacing_m'] != design['expected_origin_spacing_m']):
        raise ValueError('Measured fixture disagrees with registered layout')
    return origin
