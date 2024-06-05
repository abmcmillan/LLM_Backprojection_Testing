import numpy as np

def recon(I_p, p_angles):
    """
    Reconstruct image from sinogram I_p using filtered backprojection.

    Inputs:
    - I_p: Sinogram data, 2D array of shape (N, P) where N is the number 
           of samples along each ray and P is the number of rays
    - p_angles: 1D array of projection angles

    Returns:
    - recon_img: Reconstructed image, 2D array of shape (N, N)
    """
    
    N = I_p.shape[0] # Number of samples along each ray
    P = I_p.shape[1] # Number of projection angles
    
    # Compute ramp filter
    f = np.linspace(-N//2, N//2, N) 
    ramp_filter = np.abs(f)
    
    # Apply filter to sinogram
    I_f = np.fft.fft(I_p, axis=0)
    I_f = np.multiply(I_f, ramp_filter) 
    I_f = np.real(np.fft.ifft(I_f, axis=0))
    
    # Backproject filtered sinogram
    recon_img = np.zeros((N,N))
    for i in range(P):
        angle = p_angles[i]
        proj = I_f[:,i] 
        indices = np.round(N/2 + proj * np.cos(angle)).astype(int)
        recon_img[indices, np.arange(N)] += proj * np.sin(angle)
        
    return recon_img