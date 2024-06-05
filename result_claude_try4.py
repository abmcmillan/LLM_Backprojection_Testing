import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d

def recon(I_p, p_angles):

    # I_p is the sinogram data with shape (N, P)
    # N is the number of radial samples
    # P is the number of projection angles
    
    N, P = I_p.shape

    # p_angles are the projection angles
    p_angles = p_angles.ravel()

    # Determine image parameters
    max_angle = np.max(p_angles)
    min_angle = np.min(p_angles)
    ang_range = max_angle - min_angle

    # Assume reconstructed image is square
    img_size = int(np.floor(np.sqrt(N)))

    # Initialize empty image
    recon = np.zeros((img_size, img_size))

    # Filter projections
    I_p_filt = ndimage.gaussian_filter1d(I_p, sigma=1.0, axis=0)

    # Backproject filtered projections        
    for i in range(P):
        angle = p_angles[i]
        rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        
        # Reshape to 2D to allow broadcasting
        proj = ndimage.affine_transform(I_p_filt[:,i].reshape(N,1), matrix=rot_matrix, offset=(-img_size/2, -img_size/2))
        
        # Reshape recon to match proj for broadcasting
        recon_i = recon.reshape(1, img_size, img_size)
        
        # Add to reconstruction
        recon += proj + recon_i

    recon = recon / P

    return recon