import numpy as np


def recon(I_p, p_angles):
  """
  Reconstructs an image from its sinogram data using filtered backprojection.

  Args:
    I_p: A 2D NumPy array containing the sinogram data.
    p_angles: A 1D NumPy array containing the projection angles.

  Returns:
    A 2D NumPy array containing the reconstructed image.
  """

  # Create a ramp filter.
  ramp_filter = np.arange(I_p.shape[0])

  # Add a new dimension to the ramp filter with size 1.
  ramp_filter = ramp_filter[:, np.newaxis]

  # Calculate the filtered projections.
  filtered_projections = I_p * ramp_filter

  # Backproject the filtered projections.
  reconstructed_image = np.zeros(I_p.shape[1])
  for i in range(I_p.shape[1]):
    # Add a new dimension to filtered_projections[:, i] with size 1.
    reconstructed_image += filtered_projections[:, i, np.newaxis] * np.cos(p_angles - i)

  return reconstructed_image


# Example usage:

I_p = np.random.randn(400, 256)
p_angles = np.random.rand(400)

# Reconstruct the image.
reconstructed_image = recon(I_p, p_angles)
