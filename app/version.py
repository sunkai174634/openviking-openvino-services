"""Single source of truth for the service version (semver).

All sidecars and the log viewer report this version. Release process:
bump here, tag the commit as ``vX.Y.Z``, rebuild images with the same tag.
"""

__version__ = '1.0.1'
