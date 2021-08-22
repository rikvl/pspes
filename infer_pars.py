import numpy as np


def get_sin_i_p_u(amp_opt_kmspc, amp_cov_kmspc, d_p_pc,
                  sin2_i_e, v_0_e_kms, k_p_kms, n_samples):

    amps = np.random.multivariate_normal(amp_opt_kmspc, amp_cov_kmspc,
                                         size=n_samples)
    amp_es = amps[:, 0]
    amp_ec = amps[:, 1]
    amp_ps = amps[:, 2]
    amp_pc = amps[:, 3]

    amp2_e = amp_es**2 + amp_ec**2
    amp2_p = amp_ps**2 + amp_pc**2
    cos2_chi_e = amp_es**2 / amp2_e
    cos2_chi_p = amp_ps**2 / amp2_p

    b2_e = (1. - sin2_i_e) / (1. - sin2_i_e * cos2_chi_e)
    z2 = b2_e / (amp2_e * amp2_p) * (v_0_e_kms * k_p_kms / d_p_pc)**2
    discrim = (1. + z2)**2 - 4. * cos2_chi_p * z2
    sin2_i_p = 2. * z2 / (1. + z2 + np.sqrt(discrim))
    sin_i_p = np.sqrt(sin2_i_p)

    return sin_i_p
