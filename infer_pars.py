import numpy as np

from astropy import units as u
from astropy import uncertainty as unc


def get_sin_i_p(amp_means, amp_covar, fixed_pars, d_p_fn, n_samples):

    amp_distrib = np.random.multivariate_normal(amp_means, amp_covar,
                                                size=n_samples)
    amp_es = unc.Distribution(amp_distrib[:, 0] * u.km/u.s/u.pc**0.5)
    amp_ec = unc.Distribution(amp_distrib[:, 1] * u.km/u.s/u.pc**0.5)
    amp_ps = unc.Distribution(amp_distrib[:, 2] * u.km/u.s/u.pc**0.5)
    amp_pc = unc.Distribution(amp_distrib[:, 3] * u.km/u.s/u.pc**0.5)

    d_p = d_p_fn(n_samples)

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


def get_sin_i_p_units(amp_means_kmspc, amp_covar_kmspc,
                      sin2_i_e, v_0_e_kms, k_p_kms, d_p_fn_pc, n_samples):

    amp_distrib = np.random.multivariate_normal(amp_means_kmspc,
                                                amp_covar_kmspc,
                                                size=n_samples)
    amp_es = amp_distrib[:, 0]
    amp_ec = amp_distrib[:, 1]
    amp_ps = amp_distrib[:, 2]
    amp_pc = amp_distrib[:, 3]

    d_p = d_p_fn_pc(n_samples)

    amp2_e = amp_es**2 + amp_ec**2
    amp2_p = amp_ps**2 + amp_pc**2
    cos2_chi_e = amp_es**2 / amp2_e
    cos2_chi_p = amp_ps**2 / amp2_p

    b2_e = (1. - sin2_i_e) / (1. - sin2_i_e * cos2_chi_e)
    z2 = b2_e / (amp2_e * amp2_p) * (v_0_e_kms * k_p_kms / d_p)**2
    discrim = (1. + z2)**2 - 4. * cos2_chi_p * z2
    sin2_i_p = 2. * z2 / (1. + z2 + np.sqrt(discrim))
    sin_i_p = np.sqrt(sin2_i_p)

    return sin_i_p
