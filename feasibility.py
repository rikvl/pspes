import numpy as np

from astropy import units as u
from astropy import constants as const

from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame, EarthLocation

from scipy.optimize import curve_fit

from tqdm import trange


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


def gen_d_p():

    d_p = 156.79 * u.pc

    return d_p


def gen_free_pars():

    omega_p = 207. * u.deg
    d_p = gen_d_p()
    s = 0.422157025
    xi = 134.6 * u.deg
    v_lens = -31.9 * u.km / u.s

    free_pars = {
        'omega_p':  omega_p,
        'd_p':      d_p,
        's':        s,
        'xi':       xi,
        'v_lens':   v_lens,
    }

    return free_pars


def gen_dveff_signed(free_pars, fixed_pars, sin_cos_ph, v_earth_xyz):

    psr_coord = fixed_pars['psr_coord']
    k_p = fixed_pars['k_p']
    cos_i_p = fixed_pars['cos_i_p']

    omega_p = free_pars['omega_p']
    d_p = free_pars['d_p']
    xi = free_pars['xi']
    s = free_pars['s']
    v_lens = free_pars['v_lens']

    sin_ph_p = sin_cos_ph[2, :]
    cos_ph_p = sin_cos_ph[3, :]

    sin_i_p = np.sqrt(1. - cos_i_p**2)
    sin_xi = np.sin(xi)
    cos_xi = np.cos(xi)
    delta_omega_p = xi - omega_p
    d_eff = (1. - s) / s * d_p

    mu_par = psr_coord.pm_ra_cosdec * sin_xi + psr_coord.pm_dec * cos_xi
    v_p_sys = (d_p * mu_par).to(u.km/u.s,
                                equivalencies=u.dimensionless_angles())

    v_p_orb = - k_p / sin_i_p * (np.cos(delta_omega_p) * sin_ph_p -
                                 np.sin(delta_omega_p) * cos_ph_p * cos_i_p)

    v_p = v_p_sys + v_p_orb

    v_earth = (v_earth_xyz[1] * sin_xi + v_earth_xyz[2] * cos_xi)

    v_eff = 1. / s * v_lens - ((1. - s) / s) * v_p - v_earth

    dveff_signed = v_eff / np.sqrt(d_eff)

    return dveff_signed


def gen_dveff_abs(free_pars, fixed_pars, sin_cos_ph, v_earth_xyz):

    dveff_signed = gen_dveff_signed(free_pars, fixed_pars,
                                    sin_cos_ph, v_earth_xyz)

    dveff_abs = np.abs(dveff_signed)

    return dveff_abs


def mdl_dveff_fit(sin_cos_ph, *fit_pars):

    amp_es, amp_ec, amp_ps, amp_pc, dveff_c = fit_pars

    sin_ph_e = sin_cos_ph[0, :]
    cos_ph_e = sin_cos_ph[1, :]
    sin_ph_p = sin_cos_ph[2, :]
    cos_ph_p = sin_cos_ph[3, :]

    dveff_e = amp_es * sin_ph_e - amp_ec * cos_ph_e
    dveff_p = amp_ps * sin_ph_p - amp_pc * cos_ph_p

    dveff = np.abs(dveff_e + dveff_p + dveff_c)

    return dveff


def get_i_p(popt, pcov, fixed_pars):

    amp_es, amp_ec, amp_ps, amp_pc, dveff_c = popt

    amp_e = np.sqrt(amp_es**2 + amp_ec**2) * u.km/u.s/u.pc**0.5
    amp_p = np.sqrt(amp_ps**2 + amp_pc**2) * u.km/u.s/u.pc**0.5
    chi_e = np.arctan2(amp_ec, amp_es) * u.rad
    chi_p = np.arctan2(amp_pc, amp_ps) * u.rad

    earth_pars = fixed_pars['earth_pars']
    i_e = earth_pars['i_e']
    v_0_e = earth_pars['v_0_e']

    d_p = gen_d_p()

    b2_e = (1. - np.sin(i_e)**2) / (1. - np.sin(i_e)**2 * np.cos(chi_e)**2)
    z2 = b2_e * (v_0_e * k_p / (amp_e * amp_p * d_p))**2
    cos2_chi_p = np.cos(chi_p)**2
    discrim = (1. + z2)**2 - 4. * cos2_chi_p * z2
    sin2_i_p = ((1. + z2 - np.sqrt(discrim)) / (2. * cos2_chi_p))
    sin_i_p = np.sqrt(sin2_i_p)

    return sin_i_p


def do_i_p(fixed_pars, sin_cos_ph, v_earth_xyz, dveff_err, nmc):

    sin_i_p_fit = np.zeros(nmc)

    for j in trange(nmc):

        free_pars = gen_free_pars()

        dveff = gen_dveff_abs(free_pars, fixed_pars,
                              sin_cos_ph, v_earth_xyz)

        dveff_obs = dveff + dveff_err * np.random.normal(size=len(dveff))

        init_guess = np.array([1., 1., 1., 1., 1.])

        popt, pcov = curve_fit(mdl_dveff_fit, sin_cos_ph, dveff_obs.value,
                               p0=init_guess)

        sin_i_p_fit[j] = get_i_p(popt, pcov, fixed_pars)

    return sin_i_p_fit


if __name__ == '__main__':

    print('started')

    # pulsar parameters

    psr_coord = SkyCoord('04h37m15.99744s -47d15m09.7170s',
                         pm_ra_cosdec=121.4385 * u.mas / u.yr,
                         pm_dec=-71.4754 * u.mas / u.yr)

    p_orb_p = 5.7410459 * u.day
    asini_p = 3.3667144 * const.c * u.s
    t_asc_p = Time(54501.4671, format='mjd', scale='tdb')

    k_p = 2.*np.pi * asini_p / p_orb_p

    # earth parameters

    earth_pars = get_earth_pars(psr_coord)

    # observation parameters

    obs_loc = EarthLocation('148°15′47″E', '32°59′52″S')

    np.random.seed(654321)
    nt = 2645
    dt_mean = 16.425 * u.yr / nt
    dt = np.random.random(nt) * 2. * dt_mean
    t = Time(52618., format='mjd') + dt.cumsum()

    dveff_err = 0.05 * u.km/u.s/u.pc**0.5

    # independent variables

    p_orb_e = earth_pars['p_orb_e']
    t_asc_e = earth_pars['t_asc_e']

    ph_e = ((t - t_asc_e) / p_orb_e).to(u.dimensionless_unscaled) * u.cycle
    ph_p = ((t - t_asc_p) / p_orb_p).to(u.dimensionless_unscaled) * u.cycle

    sin_cos_ph = np.array([
        np.sin(ph_e).value,
        np.cos(ph_e).value,
        np.sin(ph_p).value,
        np.cos(ph_p).value,
    ])

    psr_frame = SkyOffsetFrame(origin=psr_coord)

    v_earth_xyz = obs_loc.get_gcrs(t).transform_to(psr_frame).velocity.d_xyz

    # MC parameters

    nmc = 1000

    # gather parameters that are known and fixed

    fixed_pars = {
        'psr_coord':    psr_coord,
        'p_orb_p':      p_orb_p,
        'asini_p':      asini_p,
        't_asc_p':      t_asc_p,
        'k_p':          k_p,
        'earth_pars':   earth_pars,
    }

    # loop over inclinations

    for i_p in [60., 70.] * u.deg:

        print(f'inclination: {i_p:.1f}')

        fixed_pars['cos_i_p'] = np.cos(i_p)

        sin_i_p_fit = do_i_p(fixed_pars, sin_cos_ph, v_earth_xyz, dveff_err,
                             nmc)

        sin_i_p_avg = np.mean(sin_i_p_fit)
        sin_i_p_std = np.std(sin_i_p_fit)

        print(sin_i_p_avg)
        print(sin_i_p_std)

        print((np.arcsin(sin_i_p_avg) * u.rad).to(u.deg))
