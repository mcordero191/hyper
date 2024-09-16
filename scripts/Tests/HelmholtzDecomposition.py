import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from scipy.fft import fftn, ifftn, fftfreq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_divergence(u, v, w, dx, dy, dz):
    
    du_dx = np.gradient(u, dx, axis=0)
    dv_dy = np.gradient(v, dy, axis=1)
    dw_dz = np.gradient(w, dz, axis=2)
    
    divergence = du_dx + dv_dy + dw_dz
    
    return divergence

def compute_vorticity(u, v, w, dx, dy, dz):
    
    dv_dz = np.gradient(v, dz, axis=2)
    dw_dy = np.gradient(w, dy, axis=1)
    du_dz = np.gradient(u, dz, axis=2)
    dw_dx = np.gradient(w, dx, axis=0)
    du_dy = np.gradient(u, dy, axis=1)
    dv_dx = np.gradient(v, dx, axis=0)
    
    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy
    
    return omega_x, omega_y, omega_z

# Create finite difference matrix for the Laplacian
def create_laplacian_matrix(nx, ny, nz, dx, dy, dz):
    """
    Creates a sparse 3D Laplacian matrix with constant Neumann boundary conditions for a grid with given dimensions and grid spacing.
    
    Parameters:
        nx, ny, nz (int): Number of grid points in the x, y, and z directions.
        dx, dy, dz (float): Grid spacing in x, y, and z directions.
        zero_neumann (float): The zero value for the Neumann boundary condition.
    
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix representation of the Laplacian.
    """
    
    N = nx * ny * nz
    diagonals = []
    offsets = []

    # Main diagonal
    main_diag = np.ones(N) * (-2.0 / dx**2 - 2.0 / dy**2 - 2.0 / dz**2)
    diagonals.append(main_diag)
    offsets.append(0)

    # Off diagonals for x-direction
    for i in [-1, 1]:
        diag = np.ones(N) / dx**2
        # Apply Neumann boundary conditions by adjusting diagonals
        diag[i::nx] = 0
        diagonals.append(diag)
        offsets.append(i)

    # Off diagonals for y-direction
    for i in [-nx, nx]:
        diag = np.ones(N) / dy**2
        diag[i::nx*ny] = 0
        diagonals.append(diag)
        offsets.append(i)

    # Off diagonals for z-direction
    for i in [-nx * ny, nx * ny]:
        diag = np.ones(N) / dz**2
        diagonals.append(diag)
        offsets.append(i)

    # Create sparse matrix
    laplacian = sp.diags(diagonals, offsets, shape=(N, N), format='csr')
    
    return laplacian

def create_laplacian_3d_cte(nx, ny, nz, dx, dy, dz, constant_neumann=0.0):
    """
    Creates a sparse 3D Laplacian matrix with constant Neumann boundary conditions for a grid with given dimensions and grid spacing.
    
    Parameters:
        nx, ny, nz (int): Number of grid points in the x, y, and z directions.
        dx, dy, dz (float): Grid spacing in x, y, and z directions.
        constant_neumann (float): The constant value for the Neumann boundary condition.
    
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix representation of the Laplacian.
    """
    N = nx * ny * nz
    main_diagonal = np.zeros(N)
    lower_x_diagonal = np.zeros(N - 1)
    upper_x_diagonal = np.zeros(N - 1)
    lower_y_diagonal = np.zeros(N - nx)
    upper_y_diagonal = np.zeros(N - nx)
    lower_z_diagonal = np.zeros(N - nx * ny)
    upper_z_diagonal = np.zeros(N - nx * ny)
    
    dx2 = dx * dx
    dy2 = dy * dy
    dz2 = dz * dz

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                index = x + y * nx + z * nx * ny
                main_diagonal[index] = -2.0 / dx2 - 2.0 / dy2 - 2.0 / dz2  # Center point

                # X-direction
                if x > 0:
                    lower_x_diagonal[index - 1] = 1.0 / dx2  # Left point
                else:
                    # Neumann BC on left boundary (x = 0)
                    main_diagonal[index] += 1.0 / dx2
                    # Apply constant Neumann adjustment for left boundary
                    main_diagonal[index] += constant_neumann / dx

                if x < nx - 1:
                    upper_x_diagonal[index] = 1.0 / dx2  # Right point
                else:
                    # Neumann BC on right boundary (x = nx - 1)
                    main_diagonal[index] += 1.0 / dx2
                    # Apply constant Neumann adjustment for right boundary
                    main_diagonal[index] -= constant_neumann / dx

                # Y-direction
                if y > 0:
                    lower_y_diagonal[index - nx] = 1.0 / dy2  # Bottom point
                else:
                    # Neumann BC on bottom boundary (y = 0)
                    main_diagonal[index] += 1.0 / dy2
                    # Apply constant Neumann adjustment for bottom boundary
                    main_diagonal[index] += constant_neumann / dy
                
                if y < ny - 1:
                    upper_y_diagonal[index] = 1.0 / dy2  # Top point
                else:
                    # Neumann BC on top boundary (y = ny - 1)
                    main_diagonal[index] += 1.0 / dy2
                    # Apply constant Neumann adjustment for top boundary
                    main_diagonal[index] -= constant_neumann / dy

                # Z-direction
                if z > 0:
                    lower_z_diagonal[index - nx * ny] = 1.0 / dz2  # Back point
                else:
                    # Neumann BC on back boundary (z = 0)
                    main_diagonal[index] += 1.0 / dz2
                    # Apply constant Neumann adjustment for back boundary
                    main_diagonal[index] += constant_neumann / dz
                
                if z < nz - 1:
                    upper_z_diagonal[index] = 1.0 / dz2  # Front point
                else:
                    # Neumann BC on front boundary (z = nz - 1)
                    main_diagonal[index] += 1.0 / dz2
                    # Apply constant Neumann adjustment for front boundary
                    main_diagonal[index] -= constant_neumann / dz

    # Construct the sparse matrix with appropriate diagonals
    diagonals = [
        lower_z_diagonal, lower_y_diagonal, lower_x_diagonal,
        main_diagonal,
        upper_x_diagonal, upper_y_diagonal, upper_z_diagonal
    ]
    offsets = [-nx * ny, -nx, -1, 0, 1, nx, nx * ny]
    
    laplacian = sp.diags(diagonals, offsets, shape=(N, N), format='csr')
    
    return laplacian

def create_laplacian_3d(nx, ny, nz, dx, dy, dz):
    """
    Creates a sparse 3D Laplacian matrix with Neumann boundary conditions
    where the boundary gradient matches the gradient of the last interior point.
    
    Parameters:
        nx, ny, nz (int): Number of grid points in the x, y, and z directions.
        dx, dy, dz (float): Grid spacing in x, y, and z directions.
    
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix representation of the Laplacian.
    """
    N = nx * ny * nz
    main_diagonal = np.zeros(N)
    lower_x_diagonal = np.zeros(N - 1)
    upper_x_diagonal = np.zeros(N - 1)
    lower_y_diagonal = np.zeros(N - nx)
    upper_y_diagonal = np.zeros(N - nx)
    lower_z_diagonal = np.zeros(N - nx * ny)
    upper_z_diagonal = np.zeros(N - nx * ny)
    
    dx2 = dx * dx
    dy2 = dy * dy
    dz2 = dz * dz

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                index = x + y * nx + z * nx * ny
                
                # Set the value for the main diagonal
                main_diagonal[index] = -2.0 / dx2 - 2.0 / dy2 - 2.0 / dz2

                # X-direction
                if x > 0:
                    lower_x_diagonal[index - 1] = 1.0 / dx2  # Left point
                if x < nx - 1:
                    upper_x_diagonal[index] = 1.0 / dx2  # Right point

                # Y-direction
                if y > 0:
                    lower_y_diagonal[index - nx] = 1.0 / dy2  # Bottom point
                if y < ny - 1:
                    upper_y_diagonal[index] = 1.0 / dy2  # Top point

                # Z-direction
                if z > 0:
                    lower_z_diagonal[index - nx * ny] = 1.0 / dz2  # Back point
                if z < nz - 1:
                    upper_z_diagonal[index] = 1.0 / dz2  # Front point

                # Neumann BC on the boundaries:
                # If we're at the boundary, adjust the main diagonal
                # and treat the boundary as if it had the same gradient as the last interior point.
                
                # X boundaries
                if x == 0:
                    # Apply Neumann boundary condition at x=0
                    main_diagonal[index] += 1.0 / dx2
                    upper_x_diagonal[index] -= 1.0 / dx2
                elif x == nx - 1:
                    # Apply Neumann boundary condition at x=nx-1
                    main_diagonal[index] += 1.0 / dx2
                    lower_x_diagonal[index - 1] -= 1.0 / dx2

                # Y boundaries
                if y == 0:
                    # Apply Neumann boundary condition at y=0
                    main_diagonal[index] += 1.0 / dy2
                    upper_y_diagonal[index] -= 1.0 / dy2
                elif y == ny - 1:
                    # Apply Neumann boundary condition at y=ny-1
                    main_diagonal[index] += 1.0 / dy2
                    lower_y_diagonal[index - nx] -= 1.0 / dy2

                # Z boundaries
                if z == 0:
                    # Apply Neumann boundary condition at z=0
                    main_diagonal[index] += 1.0 / dz2
                    upper_z_diagonal[index] -= 1.0 / dz2
                elif z == nz - 1:
                    # Apply Neumann boundary condition at z=nz-1
                    main_diagonal[index] += 1.0 / dz2
                    lower_z_diagonal[index - nx * ny] -= 1.0 / dz2

    # Construct the sparse matrix with appropriate diagonals
    diagonals = [
        lower_z_diagonal, lower_y_diagonal, lower_x_diagonal,
        main_diagonal,
        upper_x_diagonal, upper_y_diagonal, upper_z_diagonal
    ]
    offsets = [-nx * ny, -nx, -1, 0, 1, nx, nx * ny]
    
    laplacian = sp.diags(diagonals, offsets, shape=(N, N), format='csr')
    
    return laplacian

def solve_poisson_fd(f, dx, dy, dz):
    
    nx, ny, nz = f.shape
    
    # Convert 3D grid to a 1D vector for matrix operations
    f_flat = f.ravel()
    
    # Create the Laplacian matrix
    L = create_laplacian_matrix(nx, ny, nz, dx, dy, dz)
    
    # Solve the linear system L * phi = -f for phi
    phi = splinalg.spsolve(L, f_flat)
    
    # Reshape solution back to 3D grid
    phi = phi.reshape((nx, ny, nz))
    
    return phi


def solve_poisson_fft(f, dx, dy, dz):
    """
    Solve Poisson's equation using FFT for the scalar or vector potential.

    Parameters:
        f : 3D numpy array (the divergence or each component of the curl)
        dx, dy, dz : Grid spacing in x, y, z directions

    Returns:
        solution : 3D numpy array, solution to the Poisson's equation
    """
    nx, ny, nz = f.shape
    
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    kz = 2 * np.pi * fftfreq(nz, dz)
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Fourier transform of the right-hand side
    F_f = fftn(f)

    # Avoid division by zero at the zero frequency component
    K2 = (KX**2 + KY**2 + KZ**2)
    K2[K2 == 0] = 1.0

    # Solve Poisson's equation in Fourier space
    F_solution = F_f / -K2
    F_solution[0, 0, 0] = 0.0

    # Inverse Fourier transform to get back to real space
    solution = np.real(ifftn(F_solution))

    return solution

def compute_components(u, v, w, dx, dy, dz):
    """
    Decompose the wind field into curl-free (irrotational) and 
    divergence-free (solenoidal) components using Helmholtz decomposition.

    Parameters:
        u, v, w : 3D numpy arrays of the wind components
        dx, dy, dz : Grid spacing in x, y, z directions

    Returns:
        V_d (curl-free component), V_r (divergence-free component)
    """
    
    # Compute the mean of the wind field components
    mean_u = np.mean(u)
    mean_v = np.mean(v)
    mean_w = np.mean(w)

    # Remove the mean to get the fluctuating components
    u_fluc = u #- mean_u
    v_fluc = v #- mean_v
    w_fluc = w #- mean_w
    
    # Step 1: Compute divergence and vorticity
    divergence = compute_divergence(u_fluc, v_fluc, w_fluc, dx, dy, dz)
    omega_x, omega_y, omega_z = compute_vorticity(u_fluc, v_fluc, w_fluc, dx, dy, dz)
    
    # Step 2: Solve Poisson's equation for the scalar potential (phi) and vector potential (A)
    phi = solve_poisson_fd(divergence, dx, dy, dz)
    
    A_x = solve_poisson_fd(-omega_x, dx, dy, dz)
    A_y = solve_poisson_fd(-omega_y, dx, dy, dz)
    A_z = solve_poisson_fd(-omega_z, dx, dy, dz)
    
    # Step 3: Compute the gradient of phi to get the curl-free component
    V_d_x = np.gradient(phi, dx, axis=0)
    V_d_y = np.gradient(phi, dy, axis=1)
    V_d_z = np.gradient(phi, dz, axis=2)
    
    # Step 4: Compute the curl of A to get the divergence-free component
    V_r_x = np.gradient(A_z, dy, axis=1) - np.gradient(A_y, dz, axis=2)
    V_r_y = np.gradient(A_x, dz, axis=2) - np.gradient(A_z, dx, axis=0)
    V_r_z = np.gradient(A_y, dx, axis=0) - np.gradient(A_x, dy, axis=1)
    
    return (mean_u, mean_v, mean_w), (V_d_x, V_d_y, V_d_z), (V_r_x, V_r_y, V_r_z)

def plot_wind_components(x, y, u, v, u_d, v_d, u_r, v_r, title_prefix='Wind Field Components', scale=4.0):
    """
    Plot the original wind field, curl-free (irrotational) component, and divergence-free (solenoidal) component.

    Parameters:
        u, v : 2D numpy arrays of the original horizontal wind components
        u_d, v_d : 2D numpy arrays of the curl-free (irrotational) horizontal wind components
        u_r, v_r : 2D numpy arrays of the divergence-free (solenoidal) horizontal wind components
        title_prefix: A string prefix for the plot titles
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original wind field
    axes[0].quiver(x, y, u, v, scale=scale, scale_units='inches')
    axes[0].set_title(f'{title_prefix}: Original Wind Field')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # Curl-free (irrotational) component
    axes[1].quiver(x, y, u_d, v_d, scale=scale, scale_units='inches')
    axes[1].set_title(f'{title_prefix}: Curl-Free Component')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')

    # Divergence-free (solenoidal) component
    axes[2].quiver(x, y, u_r, v_r, scale=scale, scale_units='inches')
    axes[2].set_title(f'{title_prefix}: Divergence-Free Component')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')

    plt.tight_layout()
    plt.show()

def plot_3d_wind_components(u, v, w, u_d, v_d, w_d, u_r, v_r, w_r, title_prefix='V'):
    """
    Plot the original wind field, curl-free (irrotational) component, and divergence-free (solenoidal) component in 3D.

    Parameters:
        u, v, w : 3D numpy arrays of the original wind components
        u_d, v_d, w_d : 3D numpy arrays of the curl-free (irrotational) wind components
        u_r, v_r, w_r : 3D numpy arrays of the divergence-free (solenoidal) wind components
        title_prefix: A string prefix for the plot titles
    """
    fig = plt.figure(figsize=(18, 6))

    # Create grid coordinates
    nx, ny, nz = u.shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Original wind field plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.quiver(X, Y, Z, u, v, w, length=0.5, normalize=True)
    ax1.set_title(f'{title_prefix}: Original Wind Field')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Curl-free (irrotational) component plot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.quiver(X, Y, Z, u_d, v_d, w_d, length=0.5, normalize=True)
    ax2.set_title(f'{title_prefix}: Curl-Free Component')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Divergence-free (solenoidal) component plot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.quiver(X, Y, Z, u_r, v_r, w_r, length=0.5, normalize=True)
    ax3.set_title(f'{title_prefix}: Divergence-Free Component')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

def wind_field(X, Y, Z, dx, dy, dz):
    
    # u = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.cos(np.pi * Z) #+ 2
    # v = -np.cos(np.pi * X) * np.sin(np.pi * Y) * np.cos(np.pi * Z) #+ 0.001
    # w = X*0 #np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    
    phi = -(Y-0.5)**2 #- (Y-0.5)**2 #+ Z**2
    
    Vd_x = np.gradient(phi, dx, axis=0)
    Vd_y = np.gradient(phi, dy, axis=1)
    Vd_z = np.gradient(phi, dz, axis=2)
    
    A_x = np.zeros_like(X)#-Y
    A_y = np.zeros_like(X)#X
    A_z = np.zeros_like(X)#Z**2#np.zeros_like(X)
    
    # Calcular el campo rotacional V_r = curl(A)
    Vr_x = np.gradient(A_z, dy, axis=1) - np.gradient(A_y, dz, axis=2)
    Vr_y = np.gradient(A_x, dz, axis=2) - np.gradient(A_z, dx, axis=0)
    Vr_z = np.gradient(A_y, dx, axis=0) - np.gradient(A_x, dy, axis=1)
    
    # Campo de viento total V = Vd + Vr
    u = Vd_x + Vr_x
    v = Vd_y + Vr_y
    w = Vd_z + Vr_z

    # Normalización para la visualización
    norm = np.sqrt(u**2 + v**2 + w**2)
    u /= norm
    v /= norm
    w /= norm


    return(u, v, w)

# Generating sample 3D wind data for demonstration purposes
# Replace these with your actual 3D wind data arrays
nx, ny, nz = 10, 10, 10  # Smaller grid for clear visualization

x = np.linspace(0, 1, nx, endpoint=False)
y = np.linspace(0, 1, ny, endpoint=False)
z = np.linspace(0, 1, nz, endpoint=False)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

dx = 1.0/nx  # Grid spacing in the x direction
dy = 1.0/ny  # Grid spacing in the y direction
dz = 1.0/nz  # Grid spacing in the z direction


# Original wind field (u, v, w)
u, v, w = wind_field(X, Y, Z, dx, dy, dz)

(u0, v0, w0), (u_d, v_d, w_d), (u_r, v_r, w_r) = compute_components(u, v, w, dx, dy, dz)

# print("Curl-Free Component (V_d):\n", curl_free_component)
# print("Divergence-Free Component (V_r):\n", divergence_free_component)

h = 8

plot_wind_components(X[:,:,h], Y[:,:,h],
                     u[:,:,h], v[:,:,h],
                     u_d[:,:,h], v_d[:,:,h],
                     u_r[:,:,h], v_r[:,:,h],
                     title_prefix="V")

# plot_3d_wind_components(u, v, w, u_d, v_d, w_d, u_r, v_r, w_r)
