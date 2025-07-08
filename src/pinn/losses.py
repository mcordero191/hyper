import tensorflow as tf

def power_spectrum(u, v, apply_window=True):
    """
    Compute the radial power spectrum of a complex 2D vector field:
    vel = u + i v, where u, v are 2D fields.

    Parameters
    ----------
    vel : tf.Tensor of shape (H, W), complex64
        Complex-valued 2D field (e.g., vel = u + i v)
    apply_window : bool
        Whether to apply a Hann window

    Returns
    -------
    ps : tf.Tensor of shape (num_bins,)
        Radially averaged power spectrum
    """
    
    vel = tf.cast(u, tf.complex64) + 1j*tf.cast(v, tf.complex64)
    
    if apply_window:
        H, W = vel.shape
        window = tf.signal.hann_window(H)[:, None] * tf.signal.hann_window(W)[None, :]
        vel = vel * tf.cast(window, tf.complex64)

    fft = tf.signal.fft2d(vel)
    fft = tf.signal.fftshift(fft)
    power = tf.abs(fft)**2

    # Radial binning
    H, W = vel.shape
    y, x = tf.meshgrid(tf.range(-H//2, H//2), tf.range(-W//2, W//2), indexing='ij')
    r = tf.sqrt(tf.cast(x**2 + y**2, tf.float32))
    r_bins = tf.cast(tf.floor(r), tf.int32)
    r_max = tf.reduce_max(r_bins) + 1

    return tf.math.unsorted_segment_mean(power, r_bins, num_segments=r_max)

def helmholtz_spectrum(u, v):
    """
    Compute divergence and rotational power spectra from a 2D vector field.

    Parameters:
    -----------
    u, v : tf.Tensor of shape (H, W), real
        Velocity components

    Returns:
    --------
    P_div : tf.Tensor
        Radial power spectrum of divergent component
    P_rot : tf.Tensor
        Radial power spectrum of rotational component
    """
    v_complex = tf.cast(u, tf.complex64) + 1j*tf.cast(v, tf.complex64)
    
    H, W = v_complex.shape
    
    # FFT components
    v_hat = tf.signal.fft2d(v_complex)
    v_hat = tf.signal.fftshift(v_hat)

    # Wavevectors
    kx = tf.signal.fftshift(tf.cast(tf.range(-W//2, W//2), tf.float32))
    ky = tf.signal.fftshift(tf.cast(tf.range(-H//2, H//2), tf.float32))
    KX, KY = tf.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2 + 1e-8  # avoid zero division

    # Decompose into parallel and perpendicular components
    v_dot_k = tf.math.real(v_hat) * KX + tf.math.imag(v_hat) * KY
    v_dot_k = tf.complex(v_dot_k, 0.0)

    # Projection onto k-hat (divergent)
    v_div = (v_dot_k * tf.complex(KX, KY)) / tf.complex(K2, 0.0)

    # Rotational component
    v_rot = v_hat - v_div

    # Compute power
    power_div = tf.abs(v_div)**2
    power_rot = tf.abs(v_rot)**2

    # Radial binning
    y, x = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
    r = tf.sqrt(tf.cast((x - W//2)**2 + (y - H//2)**2, tf.int32))
    r_bins = tf.cast(tf.floor(r), tf.int32)
    num_bins = tf.reduce_max(r_bins) + 1

    ps_div = tf.math.unsorted_segment_mean(power_div, r_bins, num_bins)
    ps_rot = tf.math.unsorted_segment_mean(power_rot, r_bins, num_bins)
    
    return ps_div, ps_rot

def spectral_loss(u, v=None, slope=-5/3, mode="standard", weight_div=1.0, weight_rot=1.0):
    """
    Spectral loss that penalizes deviation from a target power-law spectrum.

    Parameters:
    -----------
    u : tf.Tensor of shape (T, H, W)
        Scalar field (for mode='standard') or first component of vector field
    v : tf.Tensor of shape (H, W) or None
        Second vector component (used in mode='helmholtz')
    slope : float
        Target spectral slope (e.g., -5/3 for turbulence)
    mode : str
        'standard' for scalar fields, 'helmholtz' for vector decomposition
    weight_div : float
        Weight for divergent spectral loss (only for mode='helmholtz')
    weight_rot : float
        Weight for rotational spectral loss (only for mode='helmholtz')

    Returns:
    --------
    loss : tf.Tensor (scalar)
        Spectral loss value
    """

    def _normalize(ps):
        ps = ps[1:]  # skip k=0
        return ps / (tf.reduce_sum(ps) + 1e-8)

    def _target(ps, slope):
        k = tf.cast(tf.range(1, tf.shape(ps)[0]), tf.float32)
        return _normalize(k ** slope)

    T, _, _ = u.shape
    spectra = []

    for t in range(T):
        ut = u[t]  # (H, W)
        vt = v[t]
        
        if mode == "standard":
            # Scalar field spectral loss
            ps = power_spectrum(ut, vt, apply_window=True)
            ps = _normalize(ps)
            spectra.append(ps)
            
    
        elif mode == "helmholtz":
            if v is None:
                raise ValueError("Vector field required for Helmholtz spectral loss.")
            
            P_div, P_rot = helmholtz_spectrum(ut, vt)
            P_div_norm = _normalize(P_div)
            P_rot_norm = _normalize(P_rot)
            
            spectra.append((P_div_norm, P_rot_norm))

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'standard' or 'helmholtz'.")
        
    if mode == "standard":
        
        ps_mean = tf.reduce_mean(tf.stack(spectra, axis=0), axis=0)
        tgt = _target(ps_mean, slope)
        
        loss = tf.reduce_mean((tf.math.log(ps_mean + 1e-8) - tf.math.log(tgt + 1e-8)) ** 2)
        
        return loss

    elif mode == "helmholtz":
        
        ps_div_all = tf.stack([s[0] for s in spectra], axis=0)
        ps_rot_all = tf.stack([s[1] for s in spectra], axis=0)

        ps_div_mean = tf.reduce_mean(ps_div_all, axis=0)
        ps_rot_mean = tf.reduce_mean(ps_rot_all, axis=0)

        tgt = _target(ps_div_mean, slope)

        loss_div = tf.reduce_mean((tf.math.log(ps_div_mean + 1e-8) - tf.math.log(tgt + 1e-8))**2)
        loss_rot = tf.reduce_mean((tf.math.log(ps_rot_mean + 1e-8) - tf.math.log(tgt + 1e-8))**2)

        return weight_div * loss_div + weight_rot * loss_rot