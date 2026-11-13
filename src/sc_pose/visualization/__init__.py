# src/sc_pose/visualization/__init__.py
"""public calibrated silhouette and contour utilities"""
# SPDX-License-Identifier: GPL-3.0-only
# adapted from Will-Lipscomb24/CamCal notation a82d9573163da0a71c6c1d1cda02477c8292f46f

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import cad_geometry, cad_rasterization, cad_render
from .cad_geometry import RenderMaskUnavailableError, default_render_config, project_points
from .cad_render import inner_contour, mean_contour_distance, render_cad_silhouette


UPSTREAM_SOURCE = {
    'repository': 'https://github.com/Will-Lipscomb24/CamCal',
    'revision': 'a82d9573163da0a71c6c1d1cda02477c8292f46f',
    'files': {
        'cad_geometry.py': '7ac1ecb1cf0a866c2e6519818726b117a3fce67342cd818c3236079b08a9b4f5',
        'cad_render.py': 'b5a4ed40beb5e867c41e6e5296d2f45a748e09ff6fc31eb4fff10f34f9b67fee',
    },
}


def renderer_provenance():
    """
    identify installed rendering code and its copied upstream source

    Inputs:
    none

    Outputs:
    provenance (dict): provider version, source hashes and original source identity
    """
    try:
        installed   = version('sc_pose')
    except PackageNotFoundError:
        installed   = None
    modules = (cad_geometry, cad_rasterization, cad_render)
    paths   = [Path(module.__file__).resolve() for module in modules]
    return {
        'renderer_provider': 'sc_pose', 'renderer_version': installed,
        'renderer_upstream': UPSTREAM_SOURCE,
        'source_files': {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
    }


__all__ = [
    'RenderMaskUnavailableError', 'default_render_config', 'project_points',
    'inner_contour', 'mean_contour_distance', 'render_cad_silhouette', 'renderer_provenance',
]
