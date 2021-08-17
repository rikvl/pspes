import numpy as np

from astropy import units as u
from astropy import constants as const
from astropy import uncertainty as unc

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from tqdm import trange

from feasibility import feasibility
from infer_pars import get_sin_i_p_units
from utils import get_earth_pars


def d_p_prior(n_samples):

    d_p = unc.normal(156.79*u.pc, std=0.25*u.pc, n_samples=n_samples)

    return d_p


def d_p_prior_pc(n_samples):

    d_p = np.random.normal(loc=156.79, scale=0.25, size=n_samples)

    return d_p


# set random number generator seed
np.random.seed(654321)

# --- pulsar parameters ---

psr_coord = SkyCoord('04h37m15.99744s -47d15m09.7170s',
                     pm_ra_cosdec=121.4385 * u.mas / u.yr,
                     pm_dec=-71.4754 * u.mas / u.yr)

p_orb_p = 5.7410459 * u.day
asini_p = 3.3667144 * const.c * u.s
t_asc_p = Time(54501.4671, format='mjd', scale='tdb')

# --- observation parameters ---

obs_loc = EarthLocation('148°15′47″E', '32°59′52″S')

nt = 2645
dt_mean = 16.425 * u.yr / nt
dt = np.random.random(nt) * 2. * dt_mean
t_obs = Time(52618., format='mjd') + dt.cumsum()

dveff_err = 0.2 * u.km/u.s/u.pc**0.5

# --- Monte Carlo data generation and fitting ---

nmc = 1000

(cos_i_p, omega_p, d_p, s, xi, v_lens,
 amp_opt, amp_cov) = feasibility(psr_coord, p_orb_p, asini_p, t_asc_p,
                                 d_p_prior, obs_loc, t_obs, dveff_err, nmc)

# --- parameters that are known and fixed ---

# earth parameters
earth_pars = get_earth_pars(psr_coord)

# pulsar radial velocity amplitude
k_p = 2.*np.pi * asini_p / p_orb_p
k_p_kms = k_p.to_value(u.km/u.s)

i_e = earth_pars['i_e']
sin2_i_e = (np.sin(i_e)**2).value

v_0_e = earth_pars['v_0_e']
v_0_e_kms = v_0_e.to_value(u.km/u.s)

n_samples = 1000

sin_i_p_fit = np.zeros((nmc, n_samples))
for j in trange(nmc):

    sin_i_p_fit[j, :] = get_sin_i_p_units(amp_opt[j, :], amp_cov[j, :, :],
                                          sin2_i_e, v_0_e_kms, k_p_kms,
                                          d_p_prior_pc, n_samples)

percts = np.percentile(sin_i_p_fit, [16, 50, 84], axis=-1)
q = np.diff(percts, axis=0)
comb_q = q.mean(axis=0)
comb_q_perct = np.percentile(comb_q, [16, 50, 84], axis=-1)
comb_q_q = np.diff(comb_q_perct)

print('delta sin(i_p) = '
      f'{comb_q_perct[1]:.4f} [+{comb_q_q[1]:.4f}, -{comb_q_q[0]:.4f}]')

i_p_fit = (np.arcsin(sin_i_p_fit) * u.rad).to(u.deg)
i_p_percts = np.percentile(i_p_fit, [16, 50, 84], axis=-1)
i_p_q = np.diff(i_p_percts, axis=0)
i_p_comb_q = i_p_q.mean(axis=0)
i_p_comb_q_perct = np.percentile(i_p_comb_q, [16, 50, 84], axis=-1)
i_p_comb_q_q = np.diff(i_p_comb_q_perct)

print(f'delta i_p = {i_p_comb_q_perct[1].to_value(u.deg):.2f} ['
      f'+{i_p_comb_q_q[1].to_value(u.deg):.2f}, '
      f'-{i_p_comb_q_q[0].to_value(u.deg):.2f}] deg')
