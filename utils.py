import numpy as np

from astropy import units as u

from astropy.time import Time
from astropy.coordinates import SkyCoord

from scipy.special import gammainccinv


def get_earth_pars(coord_equat):
    """Return parameters describing Earth's orbit w.r.t. a position on the sky.

    Parameters
    ----------
    coord_equat : `~astropy.coordinates.SkyCoord`
        Position on the sky in celestial coordinates.

    Returns
    -------
    i_e : `~astropy.coordinates.angles.Angle`
        Earth's orbital inclination with respect to the position on the sky.
    omega_e : `~astropy.coordinates.angles.Angle`
        Earth's longitude of ascending node with respect to the position.
    t_asc_e : `~astropy.time.Time`
        Time of Earth's passage of its ascending node w.r.t the position.
    """

    p_orb_e = 1. * u.yr

    t_eqx = Time('2005-03-21 12:33', format='iso', scale='utc')

    coord_eclip = coord_equat.barycentricmeanecliptic
    ascnod_eclip = SkyCoord(lon=coord_eclip.lon - 90.*u.deg, lat=0.*u.deg,
                            frame='barycentricmeanecliptic')
    ascnod_equat = SkyCoord(ascnod_eclip).icrs

    i_e = coord_eclip.lat + 90.*u.deg
    omega_e = coord_equat.position_angle(ascnod_equat)
    t_asc_e = t_eqx + (coord_eclip.lon + 90.*u.deg).to_value(u.cycle) * p_orb_e

    return i_e, omega_e, t_asc_e


def get_phase(t, p, t_0):
    """Get phase.

    Parameters
    ----------
    t : `~astropy.time.Time`
        Times.
    p : `~astropy.units.Quantity`
        Period.
    t_0 : `~astropy.time.Time`
        Reference time.

    Returns
    -------
    ph : `~astropy.units.Quantity`
        Phase in cycles.
    """

    ph = ((t - t_0) / p).to(u.dimensionless_unscaled) * u.cycle

    return ph


def d_prior_edsd(length_scale, size):
    r"""Exponentially decreasing space density distance prior function.

    From Bailer-Jones (2015, 2018)
    https://ui.adsabs.harvard.edu/abs/2015PASP..127..994B/abstract
    https://ui.adsabs.harvard.edu/abs/2018AJ....156...58B/abstract

    This prior is given by

    .. math::

        P(r/L) = (r/L)^2 \exp( -r/L ) / 2, \qquad r/L > 0,

    where :math:`r` is distance and :math:`L` is a length scale. The prior has
    a single mode at :math:`2 L`.

    In principle, :math:`L` should vary as a function of Galactic coordinates.
    Using a fixed :math:`L` corresponds to a simplistic, isotropic prior.

    Parameters
    ----------
    length_scale : `~astropy.units.Quantity`
        The length scale :math:`L`.
    size : int or tuple of ints
        Output shape, setting the number of samples in the array.

    Returns
    -------
    distance : `~astropy.units.Quantity`
        Array of distances sampling the prior probability distribution.
    """

    uniform_var = np.random.uniform(size=size)
    distance = gammainccinv(3, uniform_var) * length_scale

    return distance
