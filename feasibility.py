import numpy as np

from astropy import units as u
from astropy import uncertainty as unc

from astropy.coordinates import SkyOffsetFrame

from scipy.optimize import curve_fit

from tqdm import trange

from utils import get_earth_pars


def gen_free_pars(d_p_fn, nmc):

    cos_i_p = unc.uniform(lower=0., upper=1., n_samples=nmc)
    omega_p = unc.uniform(lower=0., upper=360.*u.deg, n_samples=nmc)
    d_p = d_p_fn(nmc)
    s = unc.uniform(lower=0., upper=1., n_samples=nmc)
    xi = unc.uniform(lower=0.*u.deg, upper=360.*u.deg, n_samples=nmc)
    v_lens = unc.normal(0.*u.km/u.s, std=20.*u.km/u.s, n_samples=nmc)

    return (cos_i_p.distribution,
            omega_p.distribution,
            d_p.distribution,
            s.distribution,
            xi.distribution,
            v_lens.distribution)


def gen_dveff_sgn(free_pars, fixed_pars, sin_cos_ph, v_e_xyz):

    psr_coord = fixed_pars['psr_coord']
    k_p = fixed_pars['k_p']

    cos_i_p = free_pars['cos_i_p']
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

    mu_p_sys_par = psr_coord.pm_ra_cosdec * sin_xi + psr_coord.pm_dec * cos_xi
    v_p_sys = (d_p * mu_p_sys_par).to(u.km/u.s,
                                      equivalencies=u.dimensionless_angles())

    v_p_orb = - k_p / sin_i_p * (np.cos(delta_omega_p) * sin_ph_p -
                                 np.sin(delta_omega_p) * cos_ph_p * cos_i_p)

    v_p = v_p_sys + v_p_orb

    v_e = (v_e_xyz[1] * sin_xi + v_e_xyz[2] * cos_xi)

    v_eff = 1. / s * v_lens - ((1. - s) / s) * v_p - v_e

    dveff_sgn = v_eff / np.sqrt(d_eff)

    return dveff_sgn


def mdl_dveff_har(sin_cos_ph, *fit_pars):

    amp_es, amp_ec, amp_ps, amp_pc, dveff_c = fit_pars

    sin_ph_e = sin_cos_ph[0, :]
    cos_ph_e = sin_cos_ph[1, :]
    sin_ph_p = sin_cos_ph[2, :]
    cos_ph_p = sin_cos_ph[3, :]

    dveff_e = amp_es * sin_ph_e - amp_ec * cos_ph_e
    dveff_p = amp_ps * sin_ph_p - amp_pc * cos_ph_p

    dveff = dveff_e + dveff_p + dveff_c

    return dveff


def fit_dveff(sin_cos_ph, dveff_obs):

    # make initial guess for fitting
    p0 = np.array([np.std(dveff_obs.value),
                  np.std(dveff_obs.value),
                  np.std(dveff_obs.value),
                  np.std(dveff_obs.value),
                  np.mean(dveff_obs.value)])

    # fit model of harmonic coefficients to data
    popt, pcov = curve_fit(mdl_dveff_har, sin_cos_ph, dveff_obs.value, p0=p0)

    return popt, pcov


def feasibility(psr_coord, p_orb_p, asini_p, t_asc_p, d_p_fn,
                obs_loc, t_obs, dveff_err, nmc):

    print('Started!')

    # --- parameters that are known and fixed ---

    # pulsar radial velocity amplitude
    k_p = 2.*np.pi * asini_p / p_orb_p

    # earth parameters
    earth_pars = get_earth_pars(psr_coord)

    # gather parameters into dict
    fixed_pars = {
        'psr_coord':    psr_coord,
        'p_orb_p':      p_orb_p,
        'asini_p':      asini_p,
        't_asc_p':      t_asc_p,
        'k_p':          k_p,
        'earth_pars':   earth_pars,
    }

    # --- independent variables ---

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

    # generate distributions of free parameters
    cos_i_p, omega_p, d_p, s, xi, v_lens = gen_free_pars(d_p_fn, nmc)

    # Monte Carlo data generation and fitting
    nfc = 5
    popt = np.zeros((nmc, nfc))
    pcov = np.zeros((nmc, nfc, nfc))

    for j in trange(nmc):

        free_pars = {
            'cos_i_p':  cos_i_p[j],
            'omega_p':  omega_p[j],
            'd_p':      d_p[j],
            's':        s[j],
            'xi':       xi[j],
            'v_lens':   v_lens[j],
        }

        # generate data
        dveff = gen_dveff_sgn(free_pars, fixed_pars, sin_cos_ph, v_e_xyz)

        # add noise
        dveff_obs = dveff + dveff_err * np.random.normal(size=len(dveff))

        # fit model to data
        popt[j, :], pcov[j, :, :] = fit_dveff(sin_cos_ph, dveff_obs)

    return cos_i_p, omega_p, d_p, s, xi, v_lens, popt, pcov
