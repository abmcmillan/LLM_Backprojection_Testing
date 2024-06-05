import numpy as np

def recon(I_p, p_angles):

    N = I_p.shape[0] 
    P = I_p.shape[1]

    f = np.linspace(-N//2, N//2, N)
    ramp_filter = np.abs(f)

    # Extend 1D filter to 2D
    ramp_filter = np.tile(ramp_filter[:, np.newaxis], (1, P))
    
    I_f = np.fft.fft(I_p, axis=0)
    I_f = np.multiply(I_f, ramp_filter)
    I_f = np.real(np.fft.ifft(I_f, axis=0))

    recon_img = np.zeros((N,N))
    for i in range(P):
        angle = p_angles[i]
        proj = I_f[:,i]
        indices = np.round(N/2 + proj * np.cos(angle)).astype(int)
        indices = np.clip(indices, 0, N-1)
        recon_img[indices, np.arange(N)] += proj * np.sin(angle)
        
    return recon_img