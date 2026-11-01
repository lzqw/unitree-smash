"""Preserve explicit support assistance across training, export and evaluation."""
from .centroidal_support_v1 import settings, SupportAssistedController


def policy_support_config(metadata, override=None, disabled=False):
    if disabled and override is not None:
        raise ValueError('Cannot disable support and supply a support override')
    inherited = settings(metadata.get('support_assistance'))
    if disabled:
        return None
    return settings(override) if override is not None else inherited


def add_support_controller(source, count, device, config, asset_sha256):
    if config is None:
        return source
    return SupportAssistedController(source, count, device, config, asset_sha256)
