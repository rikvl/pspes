import numpy as np

from astropy import units as u

from astropy.time import Time
from astropy.coordinates import SkyCoord


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


def d_p_prior_fb(coord_equat, size):
    """Fallback pulsar distance prior probability function.

    Placeholder for now, to be replaced.

    Parameters
    ----------
    coord_equat : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of pulsar system.
    size : int or tuple of ints
        Output shape, setting the number of samples in the array.

    Returns
    -------
    d_p : `~astropy.units.Quantity`
        Array of pulsar distances sampling the prior probability distribution.
    """

    d_p = np.random.uniform(low=0.5*u.kpc, high=2.5*u.kpc, size=size)

    return d_p
