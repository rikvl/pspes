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
    """Generate time series of signed scaled effective velocity.

    Parameters
    ----------
    cos_i_p : `~astropy.units.Quantity`
        Cosine of the pulsar system's orbital inclination.
    omega_p : `~astropy.units.Quantity`
        Pulsar's longitude of ascending node.
    d_p : `~astropy.units.Quantity`
        Pulsar's distance.
    s : `~astropy.units.Quantity`
        Fractional screen-pulsar distance
    xi : `~astropy.units.Quantity`
        Position angle of screen's line of lensed images.
    v_lens : `~astropy.units.Quantity`
        Lens's velocity component parallel to `xi`.
    k_p : `~astropy.units.Quantity`
        Pulsar's radial velocity amplitude.
    pm_ra_cosdec, pm_dec : `~astropy.units.Quantity`
        Pulsar system's proper motion components.
    sin_cos_ph : ndarray
        2D array of sines and cosines of Earth's and the pulsar's orbital
        phases at the observation times.
    v_e_xyz : `~astropy.units.Quantity`
        Observatory site's (Earth's) XYZ velocities w.r.t. the Solar System
        barycentre, with X, Y, and Z giving the velocities in the radial, RA,
        and DEC directions, respectively.

    Returns
    -------
    dveff_sgn : `~astropy.units.Quantity`
        Time series of signed scaled effective velocities.
    """

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
    """Model signed scaled effective velocities with 5 harmonic coefficients.

    Parameters
    ----------
    sin_cos_ph : ndarray
        2D array of sines and cosines of Earth's and the pulsar's orbital
        phases at the observation times.
    *fit_pars : tuple
        The five harmonic coefficients.

    Returns
    -------
    dveff : `~astropy.units.Quantity`
        Time series of signed scaled effective velocities.
    """

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
    """Fit `mdl_dveff_har` to `dveff_obs` using `scipy.optimize.curve_fit`.
    """

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
    """Pulsar system that is targetted by scintillometry.
    """

    def __init__(self, coord, p_orb_p, asini_p, t_asc_p, d_p_pc_prior_fn):
        """Initializes `TargetPulsar`.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Celestial coordinates and proper motion of the target system.
        p_orb_p : `~astropy.units.Quantity`
            Pulsar binary system's orbital period.
        asini_p : `~astropy.units.Quantity`
            Pulsar's projected semi-major axis.
        t_asc_p : `~astropy.time.Time`
            Pulsar's time of ascending node.
        d_p_pc_prior_fn : callable
            Function that returns a `~numpy.ndarray` of pulsar distances in pc
            following the prior probability distribution. Its only argument
            should be the number (`int`) of samples in the array.
        """
        self.coord = coord
        self.p_orb_p = p_orb_p
        self.asini_p = asini_p
        self.t_asc_p = t_asc_p
        self.d_p_pc_prior_fn = d_p_pc_prior_fn

        self.k_p = 2. * np.pi * self.asini_p / self.p_orb_p

        i_e, omega_e, t_asc_e = get_earth_pars(self.coord)
        self.i_e = i_e
        self.omega_e = omega_e
        self.t_asc_e = t_asc_e

    def __repr__(self):
        return (f"TargetPulsar\n"
                f"coord:\n{self.coord.__repr__()}\n"
                f"p_orb_p:\n{self.p_orb_p.__repr__()}\n"
                f"asini_p:\n{self.asini_p.__repr__()}\n"
                f"t_asc_p:\n{self.t_asc_p.__repr__()}\n"
                f"d_p_pc_prior_fn:\n{self.d_p_pc_prior_fn.__repr__()}\n")


class ObservationCampaign(object):
    """Scintillometric observation campaign
    """

    def __init__(self, obs_loc, t_obs, dveff_err):
        r"""Initializes `ObservationCampaign`.

        Parameters
        ----------
        obs_loc : `~astropy.coordinates.EarthLocation`
            Location of the observatory.
        t_obs : `~astropy.time.Time`
            Times of the observations.
        dveff_err : `~astropy.units.Quantity`
            Typical uncertainty in "scaled effective velocity" :math:`\left|
            v_\mathrm{eff,\parallel} \right| / \sqrt{ d_\mathrm{eff} }`
        """
        self.obs_loc = obs_loc
        self.t_obs = t_obs
        self.dveff_err = dveff_err

    def __repr__(self):
        return (f"ObservationCampaign\n"
                f"obs_loc:\n{self.obs_loc.__repr__()}\n"
                f"t_obs:\n{self.t_obs.__repr__()}\n"
                f"dveff_err:\n{self.dveff_err.__repr__()}\n")


class MCSimulation(object):
    r"""Monte Carlo simulation of fitting scintillometric data time series.

    Generate `nmc` instances of a scintillometric system (i.e., a pulsar-lens-
    Earth system) corresponding to a specified pulsar in a binary system.
    For each instance, the six unknown parameters of interest
    (:math:`i_\mathrm{p}`, :math:`\Omega_\mathrm{p}`, :math:`d_\mathrm{p}`,
    :math:`s`, :math:`\xi`, :math:`v_\mathrm{lens,\parallel}`) are assigned
    random values according to their prior probability distributions.
    For each instance, generate a time series of "scaled effective velocities"
    :math:`\left| v_\mathrm{eff,\parallel} \right| / \sqrt{ d_\mathrm{eff} }`,
    with added artificial noise, according to a specified scintillometric
    observation campaign. Next, fit each noisy synthetic data set with a model
    and infer the parameters of interest and their uncertainties.
    For each parameter, the distribution of the inferred uncertainties then
    informs how accurately this parameter can be measured for the specified
    pulsar system and observation campaign.
    """

    p_orb_e = 1. * u.yr
    a_e = 1. * u.au
    v_0_e = 2. * np.pi * a_e / p_orb_e

    def __init__(self, target, obs_camp):
        """Initializes `MCSimulation`.

        Parameters
        ----------
        target : `TargetPulsar`
            Pulsar system that is the target of the observation campaign.
        obs_camp : `ObservationCampaign`
            Scintillometric observation campaign to be simulated.
        """
        self.target = target
        self.obs_camp = obs_camp

        # --- independent variables ---

        self.ph_e = get_phase(self.obs_camp.t_obs,
                              MCSimulation.p_orb_e,
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

    def __repr__(self) -> str:
        return (f"MCSimulation\n\n"
                f"target:\n{self.target.__repr__()}\n"
                f"obs_camp:\n{self.obs_camp.__repr__()}\n")

    def run_mc_sim(self, nmc=1000, cos_i_p=None, omega_p=None, d_p=None,
                   s=None, xi=None, v_lens=None):
        """Run the Monte Carlo simulation.

        Parameters
        ----------
        nmc : int, default 1000
            Number of hypothetical systems to simulate.
        cos_i_p : `~astropy.units.Quantity`, optional
            Input samples's cosine of the pulsar system's orbital inclination.
            Defaults to a flat prior between -1 and 1.
        omega_p : `~astropy.units.Quantity`, optional
            Input samples's pulsar's longitude of ascending node. Defaults to a
            flat prior between 0 and 360 deg.
        d_p : `~astropy.units.Quantity`, optional
            Input samples's pulsar distance. Defaults to the distance prior
            provided by the `target` attribute.
        s : `~astropy.units.Quantity`, optional
            Input samples's fractional screen-pulsar distance. Defaults to a
            flat prior between 0 and 1.
        xi : `~astropy.units.Quantity`, optional
            Input samples's position angle of screen's line of lensed images.
            Defaults to a flat prior between 0 and 360 deg.
        v_lens : `~astropy.units.Quantity`, optional
            Input samples's lens velocity parallel to `xi`. Defaults to a
            Gaussian prior with mean 0 and std 20 km/s.
        """
        self.nmc = nmc

        # --- generate free parameter distributions from priors ---

        if cos_i_p is None:
            self.cos_i_p = unc.uniform(lower=-1., upper=1.,
                                       n_samples=self.nmc).distribution
        else:
            self.cos_i_p = cos_i_p

        if omega_p is None:
            self.omega_p = unc.uniform(lower=0.*u.deg, upper=360.*u.deg,
                                       n_samples=self.nmc).distribution
        else:
            self.omega_p = omega_p

        if d_p is None:
            self.d_p = self.target.d_p_pc_prior_fn(self.nmc) * u.pc
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

    def infer_sin_i_p(self, n_samples=1000):
        """Infer sine of pulsar orbital inclination for all MC samples.

        Parameters
        ----------
        n_samples : int, default 1000
            Number of samples to use in Monte Carlo error propagation.
        """

        # convert Quantities to floats in expected units
        k_p_kms = self.target.k_p.to_value(u.km/u.s)
        sin2_i_e = (np.sin(self.target.i_e)**2).value
        v_0_e_kms = MCSimulation.v_0_e.to_value(u.km/u.s)

        # find posterior distribution of sin_i_p for each sample
        self.sin_i_p_fit = np.zeros((self.nmc, n_samples))
        for j in trange(self.nmc):

            self.sin_i_p_fit[j, :] = get_sin_i_p_u(self.amp_opt_kmspc[j, :],
                                                   self.amp_cov_kmspc[j, :, :],
                                                   sin2_i_e,
                                                   v_0_e_kms,
                                                   k_p_kms,
                                                   self.target.d_p_pc_prior_fn,
                                                   n_samples)
