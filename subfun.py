#  @file subfun.py
#
#  SUBFUNCTIONS OF THE WORK OF 
#  ""
#
#  This .py-file is based on work of F. M. Ngole Mboula, 
#  "F. M. Ngole Mboula et al. Super-resolution method using sparse regularization for point-spread function recovery" 
#  and has close relation to the work of Samuel Farrens
#  "Samuel Farrens et al. Space variant deconvolution of galaxy survey images"

#  @author Hao Shan
#  @version 1.0
#  @date 2017.1.29
#






import numpy as np
from scipy.signal import fftconvolve
from scipy import linalg
#from astropy.io import fits
import pyfits as fits
import numpy.ma as npma
from numpy import *
import wavelet3




##
#  Function that obatins the ellipticity components. 
#
#  @param[in] im: We will obtain the ellipticity components of this image.
#  @param[in] offset: The displacement used to move the object to the image center.

#
#  @return the center point, projections of the image to the shape projection matrices, and the ellipticity components.
#  Author: F. M. Ngole Mboula



def mk_ellipticity_atoms(im,offset):
    
    mat_proj,normv = proj_mat(im.shape,offset)
    U = np.zeros((6,1))
    i = 0
    for i in range(0,6):U[i,0] = (im*mat_proj[:,:,i]).sum()
    
    e1 = U[4,0]*U[2,0]-U[0,0]**2+U[1,0]**2
    e2 = U[3,0]*U[2,0]-U[0,0]**2-U[1,0]**2
    e3 = 2*(U[5,0]*U[2,0]-U[0,0]*U[1,0])
    ell = np.zeros((1,2))
    ell[0,0] = e1/e2
    ell[0,1] = e3/e2
    centroid = np.zeros((1,2))
    centroid[0,0] = (im*mat_proj[:,:,0]).sum()/(im*mat_proj[:,:,2]).sum()
    centroid[0,1] = (im*mat_proj[:,:,1]).sum()/(im*mat_proj[:,:,2]).sum()
    return centroid,U,ell




##
#  Function that obatins the shape projection matrices. 
#
#  @param[in] siz: The size of the matrices.
#  @param[in] offset: The displacement used to move the object to the image center.

#
#  @return the shape projection matrices, and the l2 norms of the matrices.
#  Author: F. M. Ngole Mboula


def proj_mat(siz,offset=0):
    
    
    n1,n2= siz[0],siz[1]
    mat_proj = np.zeros((n1,n2,6))
    normv = np.zeros((6,))
    rx = array(range(0,n1))
    mat_proj[:,:,0] = npma.outerproduct(rx+offset,ones(n2))
    normv[0] = np.sqrt((mat_proj[:,:,0]**2).sum())
    ry = array(range(0,n2))
    mat_proj[:,:,1] = npma.outerproduct(np.ones(n1),ry+offset)
    normv[1] = np.sqrt((mat_proj[:,:,1]**2).sum())
    mat_proj[:,:,2] = np.ones((n1,n2))
    normv[2] = np.sqrt((mat_proj[:,:,2]**2).sum())
    mat_proj[:,:,3] = mat_proj[:,:,0]**2 + mat_proj[:,:,1]**2
    normv[3] = np.sqrt((mat_proj[:,:,3]**2).sum())
    mat_proj[:,:,4] = mat_proj[:,:,0]**2 - mat_proj[:,:,1]**2
    normv[4] = np.sqrt((mat_proj[:,:,4]**2).sum())
    mat_proj[:,:,5] = mat_proj[:,:,0]*mat_proj[:,:,1]
    normv[5] = np.sqrt((mat_proj[:,:,5]**2).sum())
    return mat_proj,normv
    







##
#  Function that calculates the peak signal-to-noise ratio(PSNR) between two images. 
#
#  @param[in] img1: The original clean image.
#  @param[in] img2: The restored image.

#
#  @return the PSNR value.

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = max(   np.max(np.abs(img1)),     np.max(np.abs(img2))   )
    return 10 * math.log10( PIXEL_MAX**2 / mse )



##
#  Function that calculates the soft-thresholding for a vector. 
#
#  @param[in] vec0: The vector.
#  @param[in] lamda: The value of the thresholding.
#
#  @return the soft-thresholded vector.

def soft_thresholding(vec0,lamda):

#    return np.sign(vec0) * (np.abs(vec0) - lamda) * (np.abs(vec0) >= lamda)
    
    vec = reshape(vec0,np.product(vec0.shape))
    ST_vec = np.zeros( (np.shape(vec)) )
    for i in range( 0,  len(vec)  ):
#        ST_vec[i] =  max(1 - lamda / (np.abs( vec[i] )), 0) * vec[i]          the fautal reason of np.max, max,not the same, max can only handle a single number, int or float 
       
        if vec[i]>=0:
           a = vec[i]
           if a  >=  lamda:
               ST_vec[i] =   a - lamda 
           else:
               ST_vec[i] =  0 
        else:
           a = vec[i]
           if a  <=  -lamda:
               ST_vec[i] =   a + lamda 
           else:
               ST_vec[i] =  0      
    return reshape(ST_vec,vec0.shape)




##
#  Function that implements a thresholding operation for a set of coefficients with a given data size. 
#
#  @param[in] dictionary: It is the combined dictionary Phi^(c).
#  @param[in] coef: The coefficients in the transform domain by dictionary.
#  @param[in] ndev_est_hat: The estimated noise level on the first starlet scale.
#  @param[in] tau: The constant tau. In this paper, we choose tau  as 2.5.
#
#  @return the thresholded coefficients with the same size of the input.
#  


def coef_threshold(dictionary,coef,ndev_est_hat,tau):   
    siz = coef.shape   
    coef_thr = np.zeros( (siz[0], siz[1], siz[2], siz[3]) )   
    ndev_esti = np.zeros( (siz[0], siz[1]) )   
    lamda = np.zeros( (siz[0], siz[1]) )
    
    for i in range( 0,  siz[0]  ):
        for j in range( 0,  siz[1]  ):
            
            ndev_esti[i,j] = ndev_est_hat   *    np.sqrt(sum(dictionary[i,j,:,:]**2))
            lamda[i,j] = tau * ndev_esti[i,j] * mu_gen(dictionary)[i,j]
            coef_vec = np.reshape(coef[i,j,:,:], (1,siz[2]*siz[3]))
            ST_coef = soft_thresholding(coef_vec,lamda[i,j])
            coef_thr[i,j] = np.reshape(ST_coef, (siz[2],siz[3]))
            
    return coef_thr





##
#  Function that implements a thresholding strategy, which is decided by the coefficients on 
#  the first starlet scale and the dictionary kernels. See Sect. 4.2
#
#  @param[in] x: The restored output in every iteration..
#  @param[in] fils: The wavelet filters which are calculated by:
#  filters,coarse = wavelet.get_mr_filters([nim,nim], levels=scale, opt=['-t 2','-n scale'], course=True)
#  "scale" = "sigynum" in the sub-function "directional_MexicanHat".
#  "get_mr_filters" is a sub-function in the code "wavelet"
#
#  @return the thresholded coefficients with the same size of the input.


def thresholding_strategy(x,fils):
    otho_coef = wavelet3.convolve_mr_filters(x, fils)
    abs_coef = np.abs(  otho_coef[0,:,:] - np.median(otho_coef[0,:,:])  )
    ndev_est = np.median(abs_coef)/0.6745
    L2norm = np.sqrt(sum(fils[0,:,:]**2))
    ndev_est_hat = ndev_est/L2norm
    return ndev_est_hat



    


##
#  Function that constructs the 2-D directional wavelet. It is a combination of two 1-D parts--
#  a Mexican Hat wavelet part and a smooth Gaussian part--in two orthogonal axes. sigma_0 can
#  be adjusted and we set it to be 0.2 in this code. When it is chosen, it should be a constant. 
#  Then we use sigma as a variable.
#
#  @param[in] sigynum: The combined dictionary.
#  @param[in] anglenum: restored output in every iteration.
#  @param[in] points: The noisy and blurred input in every iteration.
#  @param[in] sig0: The scale parameter of the Gaussian part.
#
#  @return the dictionary of the directional wavelet.
#  


def directional_MexicanHat(sigynum, anglenum, points, sig0):
#generate 2D directional morlet wavelet in different scales(decided by sigynum) and different 
    angle = pi/anglenum
# fun1 is the Gaussian part.
    fun1 = lambda x:  np.exp( -x**2 / (2 * sig0**2) )
# fun2 is the Mexican Hat wavelet, and it is also the two oreder derivative of fun1.
    fun2 = lambda x:   (2 / (np.sqrt(3 * sig0) * (np.pi**0.25)))  *  (1- x**2 / sig0**2)  *  np.exp( -x**2 / (2 * sig0**2) )

    F_xy = np.zeros((sigynum, anglenum,points,points))
    disp = (points - 1.0) / 2
    for isigy in range(0,sigynum):
        for itheta in range(0,anglenum): 
            for i in np.arange(0, points)- disp:
                for j in np.arange(0, points)- disp:
#                F_xy[i,j] = f1(i) *  f2(j/sigy)  
                    F_xy[isigy, itheta,i-disp,j-disp] = fun1( i*cos(angle*itheta)+j*sin(angle*itheta) ) * fun2(  (-i*sin(angle*itheta) + j*cos(angle*itheta))/ (isigy+1)  )
    return F_xy 
    
    

##
#  Function that obatins sub-gradients and thus the gradient in Eq.(17).
#
#  @param[in] dictionary: The combined dictionary.
#  @param[in] x: restored output in every iteration.
#  @param[in] y: The noisy and blurred input in every iteration.
#  @param[in] PSF: The point spread function matrix.
#
#  @return the matrix of the sub-gradient.
#    

def grad_alpha_PSF(dictionary,x,y,PSF):  

    nim = y.shape[0]
    F_PSF = np.zeros((dictionary.shape[0],dictionary.shape[1],nim,nim))
    grad_alpha = np.zeros((dictionary.shape[0],dictionary.shape[1],nim,nim))

    for isigy in range(0,dictionary.shape[0]):
        for direc in range(0,dictionary.shape[1]):
            F_PSF[isigy,direc,:,:] = fftconvolve(   np.rot90(dictionary[isigy,direc,:,:],2), np.rot90(PSF,2),   'same')
            x_PSF = fftconvolve( x,  PSF, 'same')
            grad_alpha[isigy,direc,:,:] = fftconvolve( (x_PSF - y),  F_PSF[isigy,direc,:,:],   'same')
    return grad_alpha
    


##
#  Function that obatins sub-gradients and thus the gradient in Eq.(19) of our proposed Method 4.
#
#  @param[in] dictionary: It is the combined dictionary Phi^(c).
#  @param[in] SM: The shape matrix with the size 6*(nim*nim).
#  @param[in] x: restored output in every iteration.
#  @param[in] y: The noisy and blurred input in every iteration.
#  @param[in] ips_1: The inverse of the covariance matrix Ipsilon.
#  @param[in] PSF: The point spread function matrix.
#
#  @return the matrix of the sub-gradient.
#
def grad_alpha_shape_PSF(dictionary,SM,x,y,ips_1,PSF):
    nim = y.shape[0]
    grad_alpha = np.zeros((dictionary.shape[0],dictionary.shape[1],nim,nim))
    F_PSF = np.zeros((dictionary.shape[0],dictionary.shape[1],nim,nim))

    SMT_ips_1 = np.dot(np.transpose(SM), ips_1 )
    SMT_ips_1_SM = np.dot( SMT_ips_1, SM )
    
    for isigy in range(0, dictionary.shape[0]):
        for direc in range(0,dictionary.shape[1]):           
      
            F_PSF[isigy,direc,:,:] = fftconvolve(   np.rot90(dictionary[isigy,direc,:,:],2), np.rot90(PSF,2),   'same')
            x_PSF =    fftconvolve( x,  PSF, 'same')
            grad_alpha[isigy,direc] = fftconvolve(  F_PSF[isigy,direc],   reshape(   np.dot(SMT_ips_1_SM,   reshape(x_PSF-y, (np.product((x_PSF-y).shape),1))  ) , (nim,nim)    )   ,   'same')                            
            
    return grad_alpha                


    

    
    
    
    
##
#  Function that implements the forward transform or decomposition with the dictionary from the noisy and blurred image.
#
#  @param[in] dictionary: The dictionary.
#  @param[in] x: The noisy and blurred image.
#
#  @return the coefficients.
    
def PsiT(x, dictionary):
    nim = x.shape[0]
    coef = np.zeros((dictionary.shape[0],dictionary.shape[1],nim,nim)) 
    for isigy in range(0,dictionary.shape[0]):
        for direc in range(0,dictionary.shape[1]):
            coef[isigy,direc] = fftconvolve( x, dictionary[isigy,direc], 'same') 
    return coef

    
##
#  Function that implements the backward transform or reconstruction with the dictionary from the coefficients.
#
#  @param[in] dictionary: The dictionary.
#  @param[in] coef: coefficients in the transform domain.
#
#  @return the image.
    
def Psi(coef, dictionary):
    nim = coef[0,0].shape[0]
    recons = np.zeros((nim,nim))
    for isigy in range(0,dictionary.shape[0]):
        for direc in range(0,dictionary.shape[1]): 
            recons = recons + fftconvolve( coef[isigy,direc],   np.rot90(dictionary[isigy,direc],  2),   'same') 
    return recons
   




##
#  Function that sums up the l1 norms of all kernels at all directions and on all scales. It is the second term of
#  the denominator in Eq.(30).
#
#  @param[in] dictionary: The dictionary.
#
#  @return the sum of the l1 norms of all kernels.

    
def sum_kernel_l1(dictionary):
    sum_kern = 0
    for isigy in range(0,dictionary.shape[0]):
        for direc in range(0,dictionary.shape[1]):
            sum_kern = sum_kern + np.linalg.norm(   dictionary[isigy,direc], 1  ) 
    return sum_kern

    


##
#  Function that calculates the step size Mu in Eq.(30) with respect to every direction and every scale of the dictionary.
#
#  @param[in] dictionary: The dictionary.
#
#  @return the the step size Mu.

def mu_gen(dictionary):

    mu = np.zeros((dictionary.shape[0],dictionary.shape[1])) 
    for isigy in range(0,dictionary.shape[0]):
        for direc in range(0,dictionary.shape[1]):
            mu[isigy,direc] = 1 / (    np.linalg.norm(   dictionary[isigy,direc], 1  )  *  sum_kernel_l1(  dictionary  ))

    return mu

    
    



##
#  Function that calculates the value of the minimization of Eq.(15) for each iteration, and the three terms are also calculated individually. 
#  This value is used for the stop criterian.
#
#  @param[in] x: restored output in every iteration.
#  @param[in] y: The noisy and blurred input in every iteration.
#  @param[in] coef: The coefficients in the transform domain by the dictionary.
#  @param[in] SM: The shape matrix with the size 6*(nim*nim).
#  @param[in] PSF: The point spread function matrix.
#  @param[in] beta: The weight parameter of the gradient, which has the sub-gradient presented by Eq.(19).
#  @param[in] dictionary: The dictionary.
#
#  @return the values of the three terms and the sum of them.


def minimize_term(y,x,coef,SM,PSF,beta,dictionary,ndev_est_hat,lam):
    
    diff = y - fftconvolve( x,  PSF, 'same')
    term1 = 0.5 * np.linalg.norm(diff,  2)**2
    
    coef2 =  coef_threshold( dictionary,  coef ,  ndev_est_hat,  lam)
    term2 = np.linalg.norm(  reshape(coef, (np.product(coef.shape),1)  )    ,    1)
    
    term3 = 0.5 * beta * np.linalg.norm( np.dot(SM,   reshape(diff, (np.product(diff.shape))    )     )    ,  2)**2
    
    err = term1 + term2 + term3
    return err,term1,term2,term3


