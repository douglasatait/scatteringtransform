# scatteringtransform
SHP work on the scattering transform by Douglas Tait and supervisor Dr Joe Zuntz.

'kappa_0.6_0.05_noisy.fits' is an artificial map created by Joe Zuntz using a couple of the functions in wavelets.py with added noise.
'kappa_0.6_0.05.fits' is an artificial map created by Joe Zuntz using a couple of the functions in wavelets.py without added noise.

'scattering_transform.py' was written by Douglas Tait.
'wavelets.py' was written by Joe Zuntz.

To run the program the function "run_sim()" needs to be run with the arguements "nside" and "map"

nside can be any power of 2 but only worked up to nside = 32 in our testing.
map can be 'kappa', 'kappa_noise' or 'artificial'.

The function returns the 2nd order reduced scattering coefficients and saves the 1st/2nd order reduced scattering coefficients to .csv files.
