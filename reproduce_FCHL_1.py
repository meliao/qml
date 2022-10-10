import qml
from qml.representations import generate_fchl_acsf
import numpy as np

# Dummy coordinates for a water molecule
coordinates = np.array([[1.464, 0.707, 1.056],
                        [0.878, 1.218, 0.498],
                        [2.319, 1.126, 0.952]])

# Oxygen, Hydrogen, Hydrogen
nuclear_charges = np.array([8, 1, 1])

# Generate representations for the atoms in the water molecule
rep = generate_fchl_acsf(nuclear_charges, coordinates)

print(rep)
print(rep.shape)