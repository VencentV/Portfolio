import numpy as np
import matplotlib.pyplot as plt

# Constants
ipd = 64  # mm
theta1_deg = 6.3  # degrees
theta1_rad = np.radians(theta1_deg)
tan_theta1 = np.tan(theta1_rad)

def horopter_z(x, tan_theta1, ipd=64):
    a = tan_theta1
    b = 64
    c = x**2 - (ipd/2)**2
    A = a
    B = b
    C = a * c
    discriminant = B**2 - 4*A*C
    z_vals = np.full_like(x, np.nan)
    mask = discriminant >= 0
    sqrt_discriminant = np.sqrt(discriminant[mask])
    z_vals[mask] = (-B + sqrt_discriminant) / (2*A)
    return -z_vals

x_vals = np.linspace(-400, 400, 800)
z_vals = horopter_z(x_vals, tan_theta1)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, z_vals, label="Horopter Curve", color='blue')
plt.scatter([-32, 32], [0, 0], color='red', label='Eyes')
plt.scatter([200], [-500], color='green', label='p1')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Horopter Curve (Vergence Angle = 6.3°)")
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()
plt.axis("equal")
plt.tight_layout()
plt.show()
