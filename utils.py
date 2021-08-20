import numpy as np

from astropy import units as u

from astropy.time import Time
from astropy.coordinates import SkyCoord


def get_earth_pars(coord_equat):

    p_orb_e = 1. * u.yr
    a_e = 1. * u.au

    v_0_e = 2. * np.pi * a_e / p_orb_e

    t_eqx = Time('2005-03-21 12:33', format='iso', scale='utc')

    coord_eclip = coord_equat.barycentricmeanecliptic
    ascnod_eclip = SkyCoord(lon=coord_eclip.lon - 90.*u.deg, lat=0.*u.deg,
                            frame='barycentricmeanecliptic')
    ascnod_equat = SkyCoord(ascnod_eclip).icrs

    i_e = coord_eclip.lat + 90.*u.deg
    omega_e = coord_equat.position_angle(ascnod_equat)
    t_asc_e = t_eqx + (coord_eclip.lon + 90.*u.deg).to_value(u.cycle) * p_orb_e

    earth_pars = {
        'p_orb_e':  p_orb_e,
        'a_e':      a_e,
        'v_0_e':    v_0_e,
        'i_e':      i_e,
        'omega_e':  omega_e,
        't_asc_e':  t_asc_e,
    }

    return earth_pars


def get_phase(t, p, t_0):

    ph = ((t - t_0) / p).to(u.dimensionless_unscaled) * u.cycle

    return ph
