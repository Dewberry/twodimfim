class MissingReachError(Exception):
    """Raised when a hydrofabric reach ID could not be found."""


class DownstreamModelMisalignmentError(Exception):
    """Raised when a downstream model's divide does not intersect with the upstream model's bbox."""
