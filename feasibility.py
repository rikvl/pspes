import numpy as np

from astropy import units as u
from astropy import uncertainty as unc

from astropy.coordinates import SkyOffsetFrame

from scipy.optimize import curve_fit

from tqdm import trange

from utils import get_earth_pars, get_phase
from infer_pars import get_sin_i_p_u


def gen_dveff_sgn(cos_i_p, omega_p, d_p, s, xi, v_lens,
                  k_p, pm_ra_cosdec, pm_dec, sin_cos_ph, v_e_xyz):

    sin_ph_p = sin_cos_ph[2, :]
    cos_ph_p = sin_cos_ph[3, :]

    sin_i_p = np.sqrt(1. - cos_i_p**2)
    sin_xi = np.sin(xi)
    cos_xi = np.cos(xi)
    delta_omega_p = xi - omega_p
    d_eff = (1. - s) / s * d_p

    mu_p_sys_par = pm_ra_cosdec * sin_xi + pm_dec * cos_xi
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


class TargetPulsar(object):

    def __init__(self, coord, p_orb_p, asini_p, t_asc_p, d_p_prior_fn):
        self.coord = coord
        self.p_orb_p = p_orb_p
        self.asini_p = asini_p
        self.t_asc_p = t_asc_p
        self.d_p_prior_fn = d_p_prior_fn

        self.k_p = 2. * np.pi * self.asini_p / self.p_orb_p

        earth_pars = get_earth_pars(self.coord)
        self.p_orb_e = earth_pars['p_orb_e']
        self.a_e = earth_pars['a_e']
        self.v_0_e = earth_pars['v_0_e']
        self.i_e = earth_pars['i_e']
        self.omega_e = earth_pars['omega_e']
        self.t_asc_e = earth_pars['t_asc_e']


class ObservationCampaign(object):

    def __init__(self, obs_loc, t_obs, dveff_err):
        self.obs_loc = obs_loc
        self.t_obs = t_obs
        self.dveff_err = dveff_err


class MCSimulation(object):

    def __init__(self, target, obs_camp):
        self.target = target
        self.obs_camp = obs_camp

        # --- independent variables ---

        self.ph_e = get_phase(self.obs_camp.t_obs,
                              self.target.p_orb_e,
                              self.target.t_asc_e)
        self.ph_p = get_phase(self.obs_camp.t_obs,
                              self.target.p_orb_p,
                              self.target.t_asc_p)
        self.sin_cos_ph = np.array([
            np.sin(self.ph_e).value,
            np.cos(self.ph_e).value,
            np.sin(self.ph_p).value,
            np.cos(self.ph_p).value,
        ])

        psr_frame = SkyOffsetFrame(origin=self.target.coord)
        self.v_e_xyz = (self.obs_camp.obs_loc
                                     .get_gcrs(self.obs_camp.t_obs)
                                     .transform_to(psr_frame)
                                     .velocity
                                     .d_xyz)

    def run_mc_sim(self, nmc=1000, cos_i_p=None, omega_p=None, d_p=None,
                   s=None, xi=None, v_lens=None):
        self.nmc = nmc

        # --- generate free parameter distributions from priors ---

        if cos_i_p is None:
            self.cos_i_p = unc.uniform(lower=0., upper=1.,
                                       n_samples=self.nmc).distribution
        else:
            self.cos_i_p = cos_i_p

        if omega_p is None:
            self.omega_p = unc.uniform(lower=0.*u.deg, upper=360.*u.deg,
                                       n_samples=self.nmc).distribution
        else:
            self.omega_p = omega_p

        if d_p is None:
            self.d_p = self.target.d_p_prior_fn(self.nmc).distribution
        else:
            self.d_p = d_p

        if s is None:
            self.s = unc.uniform(lower=0., upper=1.,
                                 n_samples=self.nmc).distribution
        else:
            self.s = s

        if xi is None:
            self.xi = unc.uniform(lower=0.*u.deg, upper=360.*u.deg,
                                  n_samples=self.nmc).distribution
        else:
            self.xi = xi

        if v_lens is None:
            self.v_lens = unc.normal(0.*u.km/u.s, std=20.*u.km/u.s,
                                     n_samples=self.nmc).distribution
        else:
            self.v_lens = v_lens

        # --- Monte Carlo data generation and fitting ---

        nfc = 5
        self.amp_opt_kmspc = np.zeros((self.nmc, nfc))
        self.amp_cov_kmspc = np.zeros((self.nmc, nfc, nfc))

        for j in trange(self.nmc):

            # generate data
            dveff = gen_dveff_sgn(self.cos_i_p[j],
                                  self.omega_p[j],
                                  self.d_p[j],
                                  self.s[j],
                                  self.xi[j],
                                  self.v_lens[j],
                                  self.target.k_p,
                                  self.target.coord.pm_ra_cosdec,
                                  self.target.coord.pm_dec,
                                  self.sin_cos_ph,
                                  self.v_e_xyz)

            # add noise
            dveff_obs = dveff + (self.obs_camp.dveff_err
                                 * np.random.normal(size=len(dveff)))

            # fit model to data, store optimum solution and covariance matrix
            (self.amp_opt_kmspc[j, :],
             self.amp_cov_kmspc[j, :, :]) = fit_dveff(self.sin_cos_ph,
                                                      dveff_obs)

    def infer_sin_i_p(self, d_p_prior_pc, n_samples):

        # convert Quantities to floats in expected units
        k_p_kms = self.target.k_p.to_value(u.km/u.s)
        sin2_i_e = (np.sin(self.target.i_e)**2).value
        v_0_e_kms = self.target.v_0_e.to_value(u.km/u.s)

        # find posterior distribution of sin_i_p for each sample
        self.sin_i_p_fit = np.zeros((self.nmc, n_samples))
        for j in trange(self.nmc):

            self.sin_i_p_fit[j, :] = get_sin_i_p_u(self.amp_opt_kmspc[j, :],
                                                   self.amp_cov_kmspc[j, :, :],
                                                   sin2_i_e, v_0_e_kms,
                                                   k_p_kms, d_p_prior_pc,
                                                   n_samples)
