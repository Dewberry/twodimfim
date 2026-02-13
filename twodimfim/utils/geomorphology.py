def bieger_bankfull_width(da_sqkm: float) -> float:
    """Estimate bankfull width (m) from drainage area (km2) using Bieger et al. (2015) regression for US."""
    return 2.7 * (da_sqkm**0.352)
