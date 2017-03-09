#Author: SHAN HAO
#Software: Shape preserving deconvolution for galaxy images via sparsity, shape constraint

import wavelet3
from subfun import * 
import numpy as np
import scipy as sp
import pyfits as pfits
import matplotlib.pyplot as plt
from numpy import *
#from astropy.io import fits 
from scipy.signal import fftconvolve
import matplotlib.cm as cm



####################  Original Image ###############
##
#The original image includes
orig_img = np.load('deep_galaxy_catalog_clean.npy')
#orig_img = np.load('deep_galaxy_catalog_clean_shapeinfo.npy')
#orig_img = np.load('deep_galaxy_catalog_clean_lowshapeinfo.npy')
orig_img = orig_img[475,:,:]     

plt.figure(figsize=(16,10), frameon = False)
plt.subplot(141)
plt.imshow(orig_img, cmap=cm.jet, interpolation='nearest')

nim = 41
n1, n2 = orig_img.shape
####################  Add PSF and noise to the observations(Measurements) and Initialization ###############
PSFs = pfits.getdata('psf_euclid.fits')
PSF = PSFs[:,:,0]

plt.subplot(142)
plt.imshow(PSF, cmap=cm.jet, interpolation='nearest')
orig_img_PSF = fftconvolve( orig_img, PSF, 'same')
plt.subplot(143)
plt.imshow(orig_img_PSF, cmap=cm.jet, interpolation='nearest')


sigm = 0.003

if sigm == 0.0:
    image_n = orig_img_PSF
else:
    image_n = orig_img_PSF + np.random.normal(0, sigm, (n1, n2))
   


plt.subplot(144)
plt.imshow(image_n, cmap=cm.jet, interpolation='nearest')

 
y = image_n  
#####################  generate the directional Mexican Hat wavelet  ##############################

sigma0 = 0.2
anglenum = 6
scalesnum = 5  
F_xy = directional_MexicanHat(scalesnum, anglenum, nim, sigma0) 




#The point is that, even with the directional wavelet, the galaxy with less directional information can not be restored perfectly.
#But the ones with more directional information can be.
#So, here, we add the starlet wavelet, we set the scale to be the same as the directional filter, the starlet filters of different scale
#should be add to the final direction positions of different scales of the direcrional wavelet.
#That is to say, the new anglenum should be anglenum+1.
levels=5
filters,coarse = wavelet3.get_mr_filters([nim,nim], levels=5, opt=['-t 2','-n 5'], course=True)
anglenum = anglenum + 1

fils = np.concatenate((filters, reshape(coarse,(1,nim,nim))), axis=0)


star_para=0.00005
F_xy_new = np.concatenate(  (  F_xy, star_para*reshape(fils,(5,1,nim,nim))  ), axis=1  )
F_xy = F_xy_new



###################### calculating the shape constraints  #####################
offset = 0
mat_proj,normv = proj_mat([n1,n2],offset)
SM = np.zeros((6, np.product(mat_proj[:,:,0].shape)) )
for i in range(0,6):
# SM   # 6 * (n1 * n2 = n)
    SM[i] = np.reshape( mat_proj[:,:,i],  (np.product(mat_proj[:,:,i].shape))    ) 

###################### calculating the covariance matrix of the noise multiplied by the shape matrix #####################
ips = np.dot(SM, np.transpose(SM) )   # 6 * 6
# and its inverse matrix.
ips_1 = np.linalg.inv(ips) 
#ips_1 = np.linalg.pinv(ips) 
    
############################################################################################################################
##################################       Initialization          ###########################################################
############################################################################################################################

plt.figure(figsize=(16,10), frameon = False)
Iters = 500
move_points=np.zeros((Iters , 1, 2))
ind_t = np.zeros((Iters), dtype=np.int)
ind_x = np.zeros((Iters, nim, nim))  # dtype=np.float
err = np.zeros((Iters))
ind_t[0] = 1
ind_x[0] = 0
ndev_est_hat = thresholding_strategy(y,filters)
alpha_coef = np.zeros((Iters, scalesnum, anglenum, nim, nim))
alpha_coef_tmp = np.zeros((Iters, scalesnum, anglenum, nim, nim))


# beta is the calculated by the method of U-curve
beta = 4
lam = 3

print 'galaxies restoration: denoising and deconvolution.   psf=(41, 1)'  
print 'noise level=' +  repr(sigm) +   ';   shape constrains gradient para: beta='  +  repr(beta) + ';    starlet parameter=' + repr(star_para) 
for k in range(0, Iters-1):
    grad_alphai = grad_alpha_PSF(F_xy,ind_x[k],y,PSF) 
    grad_alphai2 = beta * grad_alpha_shape_PSF(F_xy,SM,ind_x[k],y,ips_1,PSF)
    
    for scales in range(0, scalesnum):
        for directions in range(0, anglenum):
            alpha_coef_tmp[k,scales,directions,:,:] = alpha_coef[k,scales,directions,:,:]  - mu_gen(F_xy)[scales,directions]/(1) * (  grad_alphai[scales,directions,:,:] + grad_alphai2[scales,directions,:,:]  )
    
    alpha_coef[k+1] =  coef_threshold( F_xy,  alpha_coef_tmp[k] ,  ndev_est_hat,  lam)
    
#################################    Fista   #################################  
    ind_t[k+1] = (1. + np.sqrt(1. + 4. * ind_t[k] ** 2.)) / 2.
    alpha_coef[k+1,:,:,:,:]  = alpha_coef[k,:,:,:,:]  + (1 + ( ind_t[k]-1. ) / ind_t[k+1]  )   *     (alpha_coef[k+1,:,:,:,:]  - alpha_coef[k,:,:,:,:] )
##############################################################################         
    ind_x[k+1] =  Psi(alpha_coef[k+1],F_xy)   


    centroid,prodGU,eps1 = mk_ellipticity_atoms(orig_img,offset=0)
    centroid,prodGU,eps2 = mk_ellipticity_atoms(image_n,offset=0)
    centroid,prodGU,eps3 = mk_ellipticity_atoms(ind_x[k+1],offset=0)   
    
    
# stop_criterion  
    min_k = minimize_term(y,  ind_x[k],  alpha_coef[k],  SM,  PSF,  beta,  F_xy,  ndev_est_hat,  lam)
    min_k1,term1,term2,term3 = minimize_term(y,  ind_x[k+1],  alpha_coef[k+1],  SM,  PSF,  beta,  F_xy,  ndev_est_hat,  lam)
    err[k+1] = np.abs(min_k[0] - min_k1)/min_k1
    

    
    tol = 1.0e-05
    
    if sigm == 0.0:
        tol = 1.0e-03
        
        
    if err[k+1] <= tol or (k>1 and err[k]<=5*tol and err[k+1]>err[k]):
        print '!!!break at iter  '  +  repr(k+1) + '!!!  :   error=' + repr(err[k+1])
        
        distance_err = sqrt(  (eps1[0,0] - eps3[0,0])** 2. + (eps1[0,1] - eps3[0,1])** 2.  )
        print repr(distance_err) + ' :distance of the galaxy image'
        PSNR_value = psnr(orig_img, ind_x[k+1])
        print repr(PSNR_value) + ' :PSNR value of the galaxy image' 
        print 'term1:  ' + repr(term1) + ';  term2:  ' + repr(term2) + ';  term3:  ' + repr(term3)
        break
    
        
    else:
        print 'iter  '  +  repr(k+1) + ' :   error=' + repr(err[k+1])
        
        distance_err = sqrt(  (eps1[0,0] - eps3[0,0])** 2. + (eps1[0,1] - eps3[0,1])** 2.  )
        print repr(distance_err) + ' :distance of the galaxy image'
        PSNR_value = psnr(orig_img, ind_x[k+1])
        print repr(PSNR_value) + ' :PSNR value of the galaxy image' 
        print 'term1:  ' + repr(term1) + ';  term2:  ' + repr(term2) + ';  term3:  ' + repr(term3)

  
    plt.subplot(131)
    plt.imshow(orig_img,  cmap=cm.jet, interpolation='nearest')
    plt.subplot(132)
    plt.imshow(ind_x[k+1],  cmap=cm.jet, interpolation='nearest')
    plt.subplot(133)
    plt.plot(eps1[0,0],eps1[0,1],'ro')
    plt.plot(eps2[0,0],eps2[0,1],'ko')
    plt.plot(eps3[0,0],eps3[0,1],'yo')

    
    
    if k == 0:
        plt.annotate('1', 
             xy=(eps3[0,0],eps3[0,1]),  
             xycoords='data',
             xytext=(25, 15),
             textcoords='offset points',
             size=10,
             arrowprops=dict(arrowstyle="->"))
         
  
    if (k+1) % 5 == 0:
        plt.annotate(str(k+1), 
             xy=(eps3[0,0],eps3[0,1]),  
             xycoords='data',
             xytext=(25, 15),
             textcoords='offset points',
             size=10,
             arrowprops=dict(arrowstyle="->"))
    









    
    
    
    
    













