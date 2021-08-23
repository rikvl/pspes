import numpy as np

from astropy import units as u
from astropy import constants as const

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from pspes import TargetPulsar, ObservationCampaign, MCSimulation


def d_p_prior(n_samples):

    d_p = np.random.normal(loc=156.79, scale=0.25, size=n_samples) << u.pc

    return d_p


print('Example script started.')

# set random number generator seed
np.random.seed(654321)

# --- pulsar parameters ---

psr_coord = SkyCoord('04h37m15.99744s -47d15m09.7170s',
                     pm_ra_cosdec=121.4385 * u.mas / u.yr,
                     pm_dec=-71.4754 * u.mas / u.yr)

p_orb_p = 5.7410459 * u.day
asini_p = 3.3667144 * const.c * u.s
t_asc_p = Time(54501.4671, format='mjd', scale='tdb')

target_psr = TargetPulsar(psr_coord, p_orb_p, asini_p, t_asc_p)

# --- observation parameters ---

obs_loc = EarthLocation('148°15′47″E', '32°59′52″S')

nt = 2645
dt_mean = 16.425 * u.yr / nt
dt = np.random.random(nt) * 2. * dt_mean
t_obs = Time(52618., format='mjd') + dt.cumsum()

dveff_err = 0.2 * u.km/u.s/u.pc**0.5

obs_camp = ObservationCampaign(obs_loc, t_obs, dveff_err)

# --- Monte Carlo data generation and fitting ---

mcsim = MCSimulation(target_psr, obs_camp, d_p_fn=d_p_prior)

print('Generating and fitting synthetic data:')

mcsim.run_mc_sim(nmc=1000)

print('Inferring sin(i_p) from harmonic coefficients:')

mcsim.infer_sin_i_p(n_samples=1000)

sin_i_p_perct = np.percentile(mcsim.sin_i_p_fit, [16, 50, 84], axis=1)
sin_i_p_etihw = np.diff(sin_i_p_perct[::2, :], axis=0) / 2
sin_i_p_etihw_perct = np.percentile(sin_i_p_etihw, [16, 50, 84])
sin_i_p_etihw_uncer = np.diff(sin_i_p_etihw_perct)

print('delta sin(i_p) ='
      f' {sin_i_p_etihw_perct[1]:.4f} ['
      f'+{sin_i_p_etihw_uncer[1]:.4f}, '
      f'-{sin_i_p_etihw_uncer[0]:.4f}]')

i_p_fit = (np.arcsin(mcsim.sin_i_p_fit) * u.rad).to(u.deg)
i_p_perct = np.percentile(i_p_fit, [16, 50, 84], axis=1)
i_p_etihw = np.diff(i_p_perct[::2, :], axis=0) / 2
i_p_etihw_perct = np.percentile(i_p_etihw, [16, 50, 84])
i_p_etihw_uncer = np.diff(i_p_etihw_perct)

print('delta i_p ='
      f' {i_p_etihw_perct[1].to_value(u.deg):.2f} ['
      f'+{i_p_etihw_uncer[1].to_value(u.deg):.2f}, '
      f'-{i_p_etihw_uncer[0].to_value(u.deg):.2f}]')
