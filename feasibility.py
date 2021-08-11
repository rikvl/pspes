import numpy as np

from astropy import units as u
from astropy import constants as const
from astropy import uncertainty as unc

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


def get_d_p_distrib(n_samples):

    d_p = unc.normal(156.79*u.pc, std=0.25*u.pc, n_samples=n_samples)

    return d_p


def gen_free_pars_sets(nmc):

    omega_p = unc.uniform(lower=0.*u.deg, upper=360.*u.deg, n_samples=nmc)
    d_p = get_d_p_distrib(nmc)
    s = unc.uniform(lower=0., upper=1., n_samples=nmc)
    xi = unc.uniform(lower=0.*u.deg, upper=360.*u.deg, n_samples=nmc)
    v_lens = unc.normal(0.*u.km/u.s, std=20.*u.km/u.s, n_samples=nmc)

    return (omega_p.distribution,
            d_p.distribution,
            s.distribution,
            xi.distribution,
            v_lens.distribution)


def gen_dveff_signed(free_pars, fixed_pars, sin_cos_ph, v_e_xyz):

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

    v_e = (v_e_xyz[1] * sin_xi + v_e_xyz[2] * cos_xi)

    v_eff = 1. / s * v_lens - ((1. - s) / s) * v_p - v_e

    dveff_signed = v_eff / np.sqrt(d_eff)

    return dveff_signed


def gen_dveff_abs(free_pars, fixed_pars, sin_cos_ph, v_e_xyz):

    dveff_signed = gen_dveff_signed(free_pars, fixed_pars, sin_cos_ph, v_e_xyz)

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


def get_i_p(amp_means, amp_covar, fixed_pars, n_samples):

    amp_distrib = np.random.multivariate_normal(amp_means, amp_covar,
                                                size=n_samples)
    amp_es = unc.Distribution(amp_distrib[:, 0] * u.km/u.s/u.pc**0.5)
    amp_ec = unc.Distribution(amp_distrib[:, 1] * u.km/u.s/u.pc**0.5)
    amp_ps = unc.Distribution(amp_distrib[:, 2] * u.km/u.s/u.pc**0.5)
    amp_pc = unc.Distribution(amp_distrib[:, 3] * u.km/u.s/u.pc**0.5)

    d_p = get_d_p_distrib(n_samples)

    k_p = fixed_pars['k_p']
    earth_pars = fixed_pars['earth_pars']
    i_e = earth_pars['i_e']
    v_0_e = earth_pars['v_0_e']

    amp2_e = amp_es**2 + amp_ec**2
    amp2_p = amp_ps**2 + amp_pc**2
    cos2_chi_e = amp_es**2 / amp2_e
    cos2_chi_p = amp_ps**2 / amp2_p

    b2_e = (1. - np.sin(i_e)**2) / (1. - np.sin(i_e)**2 * cos2_chi_e)
    z2 = b2_e / (amp2_e * amp2_p) * (v_0_e * k_p / d_p)**2
    discrim = (1. + z2)**2 - 4. * cos2_chi_p * z2
    sin2_i_p = 2. * z2 / (1. + z2 + np.sqrt(discrim))
    sin_i_p = np.sqrt(sin2_i_p).to(u.dimensionless_unscaled)

    return sin_i_p.distribution


def do_i_p(fixed_pars, sin_cos_ph, v_e_xyz, dveff_err, nmc, n_samples):

    omega_p_set, d_p_set, s_set, xi_set, v_lens_set = gen_free_pars_sets(nmc)

    sin_i_p_fit = np.zeros((nmc, n_samples))

    for j in trange(nmc, delay=0.2):

        free_pars = {
            'omega_p':  omega_p_set[j],
            'd_p':      d_p_set[j],
            's':        s_set[j],
            'xi':       xi_set[j],
            'v_lens':   v_lens_set[j],
        }

        dveff = gen_dveff_abs(free_pars, fixed_pars, sin_cos_ph, v_e_xyz)

        dveff_obs = dveff + dveff_err * np.random.normal(size=len(dveff))

        init_guess = np.array([1., 1., 1., 1., 1.])

        popt, pcov = curve_fit(mdl_dveff_fit, sin_cos_ph, dveff_obs.value,
                               p0=init_guess)

        sin_i_p_fit[j, :] = get_i_p(popt, pcov, fixed_pars, n_samples)

    return omega_p_set, d_p_set, s_set, xi_set, v_lens_set, sin_i_p_fit


def feasibility(psr_coord, p_orb_p, asini_p, t_asc_p, i_p_to_try,
                obs_loc, t_obs, dveff_err, nmc, n_samples):

    print('Started!')

    # pulsar parameters

    k_p = 2.*np.pi * asini_p / p_orb_p

    # earth parameters

    earth_pars = get_earth_pars(psr_coord)

    # independent variables

    p_orb_e = earth_pars['p_orb_e']
    t_asc_e = earth_pars['t_asc_e']

    ph_e = ((t_obs - t_asc_e) / p_orb_e).to(u.dimensionless_unscaled) * u.cycle
    ph_p = ((t_obs - t_asc_p) / p_orb_p).to(u.dimensionless_unscaled) * u.cycle

    sin_cos_ph = np.array([
        np.sin(ph_e).value,
        np.cos(ph_e).value,
        np.sin(ph_p).value,
        np.cos(ph_p).value,
    ])

    psr_frame = SkyOffsetFrame(origin=psr_coord)

    v_e_xyz = obs_loc.get_gcrs(t_obs).transform_to(psr_frame).velocity.d_xyz

    # gather parameters that are known and fixed

    fixed_pars = {
        'psr_coord':    psr_coord,
        'p_orb_p':      p_orb_p,
        'asini_p':      asini_p,
        't_asc_p':      t_asc_p,
        'k_p':          k_p,
        'earth_pars':   earth_pars,
    }

    # loop through i_p values to test

    ni_p = len(i_p_to_try)
    omega_p_set = np.zeros((ni_p, nmc))
    d_p_set = np.zeros((ni_p, nmc))
    s_set = np.zeros((ni_p, nmc))
    xi_set = np.zeros((ni_p, nmc))
    v_lens_set = np.zeros((ni_p, nmc))
    sin_i_p_fit = np.zeros((ni_p, nmc, n_samples))

    for idx, i_p in enumerate(i_p_to_try):

        print(f'Simulating i_p = {i_p}')

        fixed_pars['cos_i_p'] = np.cos(i_p)

        (omega_p_set[idx, :],
         d_p_set[idx, :],
         s_set[idx, :],
         xi_set[idx, :],
         v_lens_set[idx, :],
         sin_i_p_fit[idx, ...]) = do_i_p(fixed_pars, sin_cos_ph, v_e_xyz,
                                         dveff_err, nmc, n_samples)

    return omega_p_set, d_p_set, s_set, xi_set, v_lens_set, sin_i_p_fit


if __name__ == '__main__':

    # pulsar parameters

    psr_coord = SkyCoord('04h37m15.99744s -47d15m09.7170s',
                         pm_ra_cosdec=121.4385 * u.mas / u.yr,
                         pm_dec=-71.4754 * u.mas / u.yr)

    p_orb_p = 5.7410459 * u.day
    asini_p = 3.3667144 * const.c * u.s
    t_asc_p = Time(54501.4671, format='mjd', scale='tdb')

    # observation parameters

    obs_loc = EarthLocation('148°15′47″E', '32°59′52″S')

    np.random.seed(654321)
    nt = 2645
    dt_mean = 16.425 * u.yr / nt
    dt = np.random.random(nt) * 2. * dt_mean
    t_obs = Time(52618., format='mjd') + dt.cumsum()

    dveff_err = 0.05 * u.km/u.s/u.pc**0.5

    # MC parameters

    nmc = 1000
    n_samples = 1000

    i_p_to_try = [137.56] * u.deg

    (omega_p_set,
     d_p_set,
     s_set,
     xi_set,
     v_lens_set,
     sin_i_p_fit) = feasibility(psr_coord, p_orb_p, asini_p, t_asc_p,
                                i_p_to_try, obs_loc, t_obs, dveff_err,
                                nmc, n_samples)

    for idx, i_p in enumerate(i_p_to_try):

        sin_i_p_avg = np.mean(sin_i_p_fit[idx, ...])
        sin_i_p_std = np.std(sin_i_p_fit[idx, ...])

        print(f'sin(i_p) = {sin_i_p_avg:.2f} +/- {sin_i_p_std:.2f}')

        i_p_fit = (np.arcsin(sin_i_p_fit[idx, ...]) * u.rad).to(u.deg)

        i_p_avg = np.mean(i_p_fit)
        i_p_std = np.std(i_p_fit)

        print(f'i_p = {i_p_avg.to_value(u.deg):.2f}'
              f' +/- {i_p_std.to_value(u.deg):.2f} deg')
