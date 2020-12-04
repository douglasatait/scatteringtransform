"""
Spherical Morlet wavelet tools based on
https://arxiv.org/pdf/astro-ph/0609159.pdf

"""

import healpy
import numpy as np
import functools

def make_signal_maps(nside, c_ell, seed=None, P=1):
    """
    Generate two Healpix signal maps, one Gaussian, and one non-Gaussian,
    with the same power spectrum C_ell and resolution nside.

    Follows method (ii) in Rocha et al 2005, available at
    http://www.mrao.cam.ac.uk/~graca/NGsims/mnr8552[001-011].pdf

    The method generates highly non-Gaussian maps, which are useful
    for testing purposes.
    """
    lmax = c_ell.size - 1
    npix = healpy.nside2npix(nside)
    
    # make a random number generator
    rng = np.random.default_rng(seed)
    
    # make a gaussian and a non-gaussian random map
    aux_map1 = rng.normal(size=npix)
    aux_map2 = aux_map1 ** (2*P)

    # turn those maps into fourier space
    alm1 = healpy.map2alm(aux_map1, lmax=lmax)
    alm2 = healpy.map2alm(aux_map2, lmax=lmax)

    # determine the indexing into the alm
    ell, m = healpy.Alm.getlm(lmax)
    
    # the pixel area is used to normalize the random map
    Apix = healpy.nside2pixarea(nside)

    # make the random fourier space components have the
    # right power spectrum
    alm1 *= (c_ell[ell] / Apix)**0.5
    alm2 *= (c_ell[ell] / Apix / 2)**0.5

    # Convert the maps back to real space
    gaussian_map = healpy.alm2map(alm1, nside, lmax=lmax, verbose=False)
    non_gaussian_map = healpy.alm2map(alm2, nside, lmax=lmax, verbose=False)
    
    return gaussian_map, non_gaussian_map

def compute_wavelet_map(L, M, R, nside, lmax):
    """
    Compue a single wavelet as a map with parameters L, M, and R,
    map at resolution nside, annd going up to lmax, which should be
    greater than or equal to L + 5

    The hard bit here is that healpy can only cope with
    real maps, but these are complex, so we have to mess about a bit
    """
    n = healpy.Alm.getsize(lmax)
    blm = np.zeros(n, dtype=np.complex)
    clm = np.zeros(n, dtype=np.complex)
    for ell in range(lmax+1):
        for m in range(ell+1):
            idx = healpy.Alm.getidx(lmax, ell, m)
            alm  = psi_lm(ell, R, +m, L, M)
            alm_ = psi_lm(ell, R, -m, L, M)
            if m % 2 == 0:
                blm[idx] = 0.5  * (alm + alm_.conjugate())
                clm[idx] = 0.5j * (alm - alm_.conjugate())
            else:
                blm[idx] = 0.5  * (alm - alm_.conjugate())
                clm[idx] = 0.5j * (alm + alm_.conjugate())

    b = healpy.alm2map(blm, nside, verbose=False)
    c = healpy.alm2map(clm, nside, verbose=False)
    return b - 1j * c

def convolve_with_wavelet(map, L, M, R, lmax):
    """
    Convolve a map with a given wavelet of parameters
    L, M, and R.
    """
    npix = map.size
    nside = healpy.npix2nside(npix)
    
    # convert the map to Fourier space
    alm = healpy.map2alm(map, lmax=lmax)
    
    # get the Fourier components of the chosen wavelet
    blm = compute_wavelet_harmonics_zonal(L, M, R, nside, lmax).copy()

    # Get the ell values
    ell, m = healpy.Alm.getlm(lmax)
    
    # Normalize for the convolution
    blm *= np.sqrt((4 * np.pi) / (2 * ell + 1))
    
    # Multiply and then inverse Fourier transform
    result = healpy.alm2map(alm * blm, nside, verbose=False)

    return result


def wavelet_transform(map, L_values, R, lmax):
    """
    Generate the convolution of each wavelet in a set with
    a given map.  For a supplied set of L values, every valid M
    value is used.

    Returns a dictionary mapping (L, M) -> convolved map
    """
    npix = map.size
    nside = healpy.npix2nside(npix)
    
    # convert the map to Fourier space
    alm = healpy.map2alm(map, lmax=lmax)
    blms = []
    
    for L in L_values:
        for M in range(0, L+1):
            # get the Fourier components of the chosen wavelet
            blm = compute_wavelet_harmonics_zonal(L, M, R, nside, lmax)
            # Get the ell values
            ell, m = healpy.Alm.getlm(lmax)
            # Normalize for the convolution
            blm *= np.sqrt((4 * np.pi) / (2 * ell + 1))
            blms.append(blm)

    # Multiply and then inverse Fourier transform
    flms = [alm * blm for blm in blms]
    results = healpy.alm2map(flms, nside, verbose=False, pol=False)
    i = 0
    output = {}
    for L in L_values:
        for M in range(0, L+1):
            output[(L, M)] = results[i]
            i += 1
    return output

def psi_lm(l, R, m, L, M):
    # Generates the harmonic space coefficients for (l,m)
    # for the given wavelet.
    return np.sqrt((2*l+1)/8/np.pi**2) * (
        np.exp(-0.5 * ((l*R - L)**2 + (m - M)**2)) 
      - np.exp(-0.5*((l*R)**2 + L**2 + (m - M)**2))
      )

# this cache thing means the second time you call it
# it is much faster - we save the result.  This costs us
# memory though
# @functools.lru_cache(None)
def compute_wavelet_harmonics_zonal(L, M, R, nside, lmax):
    # This is used to compute wavelets with m=0, which
    # are the ones we need for convolutions
    n = healpy.Alm.getsize(lmax)
    blm = np.zeros(n, dtype=np.complex)
    for ell in range(lmax+1):
        b  = psi_lm(ell, R, 0, L, M)
        for m in range(ell+1):
            idx = healpy.Alm.getidx(lmax, ell, m)
            blm[idx] = b
    return blm
