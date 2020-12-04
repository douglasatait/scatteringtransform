import os
import time
x = time.time()
import numpy as np
import healpy
import wavelets
import matplotlib.pylab as plt



def mod_mean(maps):
    """
    This function calculates the absolute value of values in a dictionary and then finds the mean of the values for each key
    """
    
    abs_map = {}
    mean_map = {}
    
    for key in maps.keys():
        abs_map[key] = abs(maps[key])
        mean_map[key] = abs_map[key].mean()
        
    return abs_map, mean_map
        
def mean_sec_ord(maps):
    """
    This function calculates the absolute value of values in a dictionary and then finds the mean of the values for each key
    """
    
    abs_map = {}
    mean_map = {}
    
    for key in maps.keys():
        abs_map[key] = abs(maps[key])
        mean_map[key] = abs_map[key].mean()
        
    return mean_map
    
    
def coeffs(mean_maps, L_values):
    
    """
    Function to calculate the reduced coefficients for each L value. (1st order)
    """
    
    reduced_coeffs = {}

    for L in L_values:
        num = 0
        reduced_coeff = 0
        for M in range(0, L+1):
            reduced_coeff += mean_maps[L, M]
            num += 1
        reduced_coeff /= num
        reduced_coeffs[L] = reduced_coeff 

    return reduced_coeffs


def second_order_coeffs(dictionary, L_values):
    """
    Function to calculate the reduced coefficients for each L value. (2nd order)
    """
    reduced = {}
    for key, value in dictionary.items():            
        if isinstance(value, dict):
            reduced[key] = coeffs(value, L_values)

        else:
            reduced[key] = coeffs(value, L_values)
    
    
    n_ell = len(L_values)
    reduced_coeffs2 = np.zeros((n_ell, n_ell))
    count = np.zeros((n_ell, n_ell))
    for i in range(n_ell):
        L = L_values[i]
        for j in range(n_ell):
            K = L_values[j]
            for M in range(0, L+1):
                reduced_coeffs2[i,j] += reduced[L, M][K]
                count[i, j] += 1
    reduced_coeffs2 /= count
    
    return reduced_coeffs2 



def run_sim(nside, map_type):
    """
    This function performs the scattering transform and saves the data for further analysis.
    
    nside determines the resolution of the simulation.
    map_type selects whether the simulation calculates highly non-gaussian maps using the make_signal_maps() 
    funtion or uses the kappa maps created by Joe Zuntz.
    """
    ### Setting the sumber if iterations depending on the map type
    if map_type == "kappa" or map_type == "kappa_noise":
        num_iter = 2
        
    elif map_type == "artificial":
        num_iter = 11
        
    for r in range(1, num_iter): 
        print(time.ctime(int(time.time())))
        v = time.time()
        
        ### Set initial Conditions
        nside = nside
        npix = healpy.nside2npix(nside)  
        lmax = 2 * nside
        R=1


        ### Create Initial Maps
        
        ### Kappa Maps
        if map_type == "kappa":
            non_gaussian_map = healpy.read_map("kappa_0.6_0.05.fits")
            non_gaussian_map = healpy.ud_grade(non_gaussian_map, nside_out=nside)
            c_ell = healpy.anafast(non_gaussian_map)
            gaussian_map = healpy.synfast(c_ell, nside=nside)
            
        if map_type == "kappa_noise":
            non_gaussian_map = healpy.read_map("kappa_0.6_0.05_noisy.fits")
            non_gaussian_map = healpy.ud_grade(non_gaussian_map, nside_out=nside)
            c_ell = healpy.anafast(non_gaussian_map)
            gaussian_map = healpy.synfast(c_ell, nside=nside)
            
        ### Artificial Highly non-Gaussian Maps
        elif map_type == "artificial":
            ell = np.arange(lmax+1)
            c_ell = 1 / (1+ell)
            c_ell[:2] = 0
            gaussian_map, non_gaussian_map = wavelets.make_signal_maps(nside, c_ell)



        ### 1st order convolution
        L_values = np.arange(2, lmax - 5)
        gaussian_map_1 = wavelets.wavelet_transform(gaussian_map, L_values, R, lmax)
        nongaussian_map_1 = wavelets.wavelet_transform(non_gaussian_map, L_values, R, lmax) 
        
        
        ### Mod/mean of 1st-Order
        abs_gauss_1, mean_gauss_1 = mod_mean(gaussian_map_1)
        abs_nongauss_1, mean_nongauss_1 = mod_mean(nongaussian_map_1)  

        ### Mean of 2nd-Order Gaussian
        mean_gauss_2 = {}
        for key in abs_gauss_1.keys():
            mean_gauss_2[key] = mean_sec_ord(wavelets.wavelet_transform(abs_gauss_1[key], L_values, R, lmax)) 
        
        if map_type == "kappa" or map_type == "kappa_noise":
            half = time.time()
            print("The time taken to run roughly half the simulation was: " + str(half-v) + str(" seconds"))
        
        
        ### Mean of 2nd-Order non-Gaussian
        mean_nongauss_2 = {}
        for key in abs_nongauss_1.keys():
            mean_nongauss_2[key] = mean_sec_ord(wavelets.wavelet_transform(abs_nongauss_1[key], L_values, R, lmax))

        
        ### Calculating Reduced Coefficients 
        # Gaussian
        reduced_coeff_gauss0 = np.array(gaussian_map.mean())
        reduced_coeff_gauss1 = coeffs(mean_gauss_1, L_values)
        reduced_coeff_gauss2 = second_order_coeffs(mean_gauss_2, L_values)

        # Non-gaussian
        reduced_coeff_nongauss0 = non_gaussian_map.mean()
        reduced_coeff_nongauss1 = coeffs(mean_nongauss_1, L_values)
        reduced_coeff_nongauss2 = second_order_coeffs(mean_nongauss_2, L_values)


        ### Converting to Arrays
        reduced_coeff_gauss1 = np.array([reduced_coeff_gauss1[key] for key in reduced_coeff_gauss1.keys()])
        reduced_coeff_nongauss1 = np.array([reduced_coeff_nongauss1[key] for key in reduced_coeff_nongauss1.keys()])

            
        ### Saving the data
        if map_type == "artificial": 
            np.savetxt("artificial_nongauss_ord1_" + str(r) + ".csv", reduced_coeff_nongauss1, delimiter=",")
            np.savetxt("artificial_nongauss_ord2_" + str(r) + ".csv", reduced_coeff_nongauss2, delimiter=",")
            np.savetxt("artificial_gauss_ord1_" + str(r) + ".csv", reduced_coeff_gauss1, delimiter=",")
            np.savetxt("artificial_gauss_ord2_" + str(r) + ".csv", reduced_coeff_gauss2, delimiter=",")
            
        if map_type == "kappa": 
            np.savetxt("kappa_nongauss_ord1_" + str(r) + ".csv", reduced_coeff_nongauss1, delimiter=",")
            np.savetxt("kappa_nongauss_ord2_" + str(r) + ".csv", reduced_coeff_nongauss2, delimiter=",")
            np.savetxt("kappa_gauss_ord1_" + str(r) + ".csv", reduced_coeff_gauss1, delimiter=",")
            np.savetxt("kappa_gauss_ord2_" + str(r) + ".csv", reduced_coeff_gauss2, delimiter=",")
            
        if map_type == "kappa_noise": 
            np.savetxt("kappa_noise_nongauss_ord1_" + str(r) + ".csv", reduced_coeff_nongauss1, delimiter=",")
            np.savetxt("kappa_noise_nongauss_ord2_" + str(r) + ".csv", reduced_coeff_nongauss2, delimiter=",")
            np.savetxt("kappa_noise_gauss_ord1_" + str(r) + ".csv", reduced_coeff_gauss1, delimiter=",")
            np.savetxt("kappa_noise_gauss_ord2_" + str(r) + ".csv", reduced_coeff_gauss2, delimiter=",")
        
        if map_type == "artificial":
            print("The simulation is roughly " +str((r/10)*100) + str("% complete"))
            y = time.time()
            print("The time taken to run round " +str(r)  + str(": ") + str(y-v) + str(" seconds"))    
    z = time.time()
    print("The time taken to run the whole simulation was: " + str(z-x) + str(" seconds"))    
    
    return reduced_coeff_nongauss2, reduced_coeff_gauss2
