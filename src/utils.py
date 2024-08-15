import numpy as np

def cosine_law(x, y, l1, l2):
    """
    Cosine law for the IK calculation
    """
    return (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)

def pol2cart(rho, phi):
    """
    Convert polar coordinates to cartesian
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)