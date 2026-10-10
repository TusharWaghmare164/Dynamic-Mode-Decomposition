import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os
import glob



def read_vtk_array(file_path, array_name):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()
    vtk_array = data.GetPointData().GetArray(array_name)
    np_array = vtk_to_numpy(vtk_array)
    return np_array, data

# Sorted file list with proper zero-padded filenames
file = sorted(glob.glob(os.path.join("sampleData", "S_*.vtk")))

X = []

# Extract vorticity z-component from each file
for f in file:
    vorticity, _ = read_vtk_array(f, "vorticity")
    X.append(vorticity[:, 2])  # z-component

X = np.matrix(X).T
X1, X2 = X[:, :-1], X[:, 1:]

# DMD via SVD
U, S, VT = np.linalg.svd(X1, full_matrices=False)
r = 21
Ur, Sr, Vr = U[:, :r], np.diag(S[:r]), VT[:r, :].conj().T
Atilda = Ur.conj().T @ X2 @ Vr @ np.linalg.inv(Sr)
eigval, eigvec = np.linalg.eig(Atilda)
Phi = X2 @ Vr @ np.linalg.inv(Sr) @ eigvec

# Time step
dt = 0.2
logmap = np.log(eigval) / dt
St = logmap.imag / (2 * np.pi)
Sig = logmap.real

# Sort by frequency
order = np.argsort(St)
St = St[order]
eigval = eigval[order]
Phi = Phi[:, order]

# Projection coefficients
B = Phi.conj().T @ X1[:, 0]

# Plot Ritz values on unit circle
theta = np.linspace(0, 2 * np.pi, 50)
a, b = np.cos(theta), np.sin(theta)

plt.scatter(eigval.real, eigval.imag, zorder=1)
plt.plot(a, b, '--', color='k', zorder=0)
plt.grid(True)
plt.title("Ritz Values on Complex Plane")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.axis("equal")
plt.show()

# Export CSVs
pd.DataFrame(logmap.T).to_csv('omega.csv', index=False)
pd.DataFrame(Sig.T).to_csv('Sig.csv', index=False)
pd.DataFrame(St.T).to_csv('St.csv', index=False)
pd.DataFrame(eigval.T).to_csv('lamda.csv', index=False)
pd.DataFrame(B).to_csv('magnitude.csv', index=False)
pd.DataFrame(np.linalg.norm(Phi.real, axis=0)).to_csv('real_norm.csv', index=False)
pd.DataFrame(np.linalg.norm(Phi.imag, axis=0)).to_csv('imag_norm.csv', index=False)
pd.DataFrame(np.linalg.norm(Phi, axis=0)).to_csv('phi_norm.csv', index=False)
pd.DataFrame(np.linalg.norm(B, axis=-1)).to_csv('b_norm.csv', index=False)

# Read original mesh once to modify and write DMD modes
_, sample_data = read_vtk_array(file[0], "vorticity")

for i in range(r):
    # Get real part of Phi[:, i] as z-component
    Phi_real_i = Phi.real[:, i].flatten()

    # Get full vorticity array and modify z-component
    vorticity_array = sample_data.GetPointData().GetArray("vorticity")
    full_vorticity = vtk_to_numpy(vorticity_array).copy()
    full_vorticity[:, 2] = Phi_real_i  # replace z-component

    # Convert back to vtk array
    updated_vorticity = numpy_to_vtk(full_vorticity)
    updated_vorticity.SetName("vorticity")

    # Replace vorticity in mesh
    sample_data.GetPointData().RemoveArray("vorticity")
    sample_data.GetPointData().AddArray(updated_vorticity)

    # Write in VTK legacy format (v4.2)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(f"mode_{i+1:03}.vtk")
    writer.SetInputData(sample_data)
    writer.SetFileTypeToASCII()  # For readable legacy VTK format
    writer.Write()
