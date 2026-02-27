import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def sample_from_annulus(n, r, R, seed=None):
    """Sample points from a 2D annulus.

    This function samples `N` points from an annulus with inner radius `r`
    and outer radius `R`.

    Parameters
    ----------
    n : int
        Number of points to sample

    r : float
        Inner radius of annulus

    R : float
        Outer radius of annulus

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    torch.tensor of shape `(n, 2)`
        Tensor containing sampled coordinates.
    """
    if r >= R:
        raise RuntimeError(
            'Inner radius must be less than or equal to outer radius'
        )

    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2 * np.pi, n)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(rng.uniform(r ** 2, R ** 2, n))

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    return X
def sample_from_circle(num_points, radius=1, seed=3):
    circle = np.block([
    [np.random.uniform(low=-0.85*radius,high=0.85*radius,size=(int(0.05*num_points),2))],
    [sample_from_annulus(int(0.9*num_points), 0.85*radius,radius)]
    ])
    return circle 
def sample_from_noisy_circle(num_points=400, noise=0.2, radius=1.0, seed=3):
    circle = np.block([
    [np.random.uniform(low=-2*radius,high=2*radius,size=(int(noise*num_points),2))],
    [sample_from_annulus(int(num_points), 0.85*radius,radius)]
    ])
    return circle 
def plot_density(points):

    # Calculate density estimate
    kde = gaussian_kde(points.T)
    density = np.exp(-1*kde(points.T))

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=density, cmap='viridis')
    plt.colorbar(scatter, label='Density')
    plt.title('Scatter plot with a Gaussian density estimate')
