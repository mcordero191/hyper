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
    
class GradNormLoss:
    
    def __init__(self, alpha=1e-3, w_min=1e-3, w_max=1e3,
                 warmup_epochs=10, target_frac=None):
        
        self.alpha = alpha
        self.w_min = w_min
        self.w_max = w_max
        self.warmup_epochs = warmup_epochs
        self.epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # pesos iniciales
        self.w_div     = tf.Variable(w_min, trainable=False, dtype=tf.float32)
        self.w_mom     = tf.Variable(w_min, trainable=False, dtype=tf.float32)
        self.w_div_vor = tf.Variable(w_min, trainable=False, dtype=tf.float32)

        # fracciones objetivo
        self.target_frac = target_frac or {
            "div": 1.00, "mom": 0.50, "div_vor": 0.00
        }

    def __call__(self, L_data, L_div, L_div_vor, L_mom, L_srt):
        # pérdidas ponderadas
        loss_total = (
            L_data
            + self.w_div * L_div
            + self.w_div_vor * L_div_vor
            + self.w_mom * L_mom
            + L_srt
        )
        logs = {
            "loss_total": loss_total,
            "loss_data": L_data,
            "loss_div": L_div,
            "loss_div_vor": L_div_vor,
            "loss_mom": L_mom,
            "loss_srt": L_srt,
            "w_data": tf.constant(1.0),
            "w_div": self.w_div,
            "w_div_vor": self.w_div_vor,
            "w_mom": self.w_mom,
            "w_srt": tf.constant(1.0),
        }
        return loss_total, logs
    
    def update(self, grads, logs):
        self.epoch.assign_add(1)

        if self.epoch < self.warmup_epochs:
            return 0.0  # warmup → no ajustar

        # magnitudes
        g_data    = tf.abs(grads["grad_data"]    *1.0)
        g_div     = tf.abs(grads["grad_div"]     * self.w_div)
        g_mom     = tf.abs(grads["grad_mom"]     * self.w_mom)
        g_div_vor = tf.abs(grads["grad_div_vor"] * self.w_div_vor)

        g_tot = g_data + g_div + g_mom + g_div_vor + 1e-12

        # ratios GradNorm
        ratio_div     = (self.target_frac["div"]     * g_tot) / (g_div     + 1e-12)
        ratio_mom     = (self.target_frac["mom"]     * g_tot) / (g_mom     + 1e-12)
        ratio_div_vor = (self.target_frac["div_vor"] * g_tot) / (g_div_vor + 1e-12)

        def smooth_update(var, ratio):
            target_w = tf.clip_by_value(var * ratio, self.w_min, self.w_max)
            return (1.0 - self.alpha) * var + self.alpha * target_w

        self.w_div.assign(smooth_update(self.w_div, ratio_div))
        self.w_mom.assign(smooth_update(self.w_mom, ratio_mom))
        self.w_div_vor.assign(smooth_update(self.w_div_vor, ratio_div_vor))

        logs.update(dict(
            # frac_div=g_div / g_tot,
            # frac_mom=g_mom / g_tot,
            # frac_div_vor=g_div_vor / g_tot,
            w_div=self.w_div,
            w_mom=self.w_mom,
            w_div_vor=self.w_div_vor,
        ))

        return 1.0

class WeightScheduler:
    
    def __init__(self, total_epochs=5000, alpha=1e-3, warmup_epochs=10):
        
        self.epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.total_epochs = total_epochs
        self.alpha = alpha

        self.warmup_epochs = warmup_epochs
        # initial weights
        self.w_srt = 1e-4
        self.w_div = tf.Variable(1e-6, trainable=False, dtype=tf.float32)
        self.w_mom = tf.Variable(1e-6, trainable=False, dtype=tf.float32)

        # ranges
        self.div_min, self.div_max = 1e-3, 1e3
        self.mom_min, self.mom_max = 1e-3, 1e3

    def _log_schedule(self, epoch, w_min, w_max):
        # progress in [0,1]
        frac = tf.cast(epoch, tf.float32) / float(self.total_epochs)
        # exponential interpolation in log-space
        log_min, log_max = tf.math.log(w_min)/tf.math.log(10.0), tf.math.log(w_max)/tf.math.log(10.0)
        log_target = log_min + frac * (log_max - log_min)
        return tf.pow(10.0, log_target)

    def update(self, *args):
        # get target weights
        # target_div = self._log_schedule(self.epoch, self.div_min, self.div_max)
        # target_mom = self._log_schedule(self.epoch, self.mom_min, self.mom_max)
        # increment epoch
        self.epoch.assign_add(1)
        
        if self.epoch < self.warmup_epochs:
            return 0.0
        # EMA update
        new_w_div = (1.0 - self.alpha) * self.w_div + self.alpha * self.div_max
        new_w_mom = (1.0 - self.alpha) * self.w_mom + self.alpha * self.mom_max

        self.w_div.assign(new_w_div)
        self.w_mom.assign(new_w_mom)
        
        return 1.0
    
    def __call__(self, L_data, L_div, L_div_vor, L_mom, L_srt):
        # pérdidas ponderadas
        loss_total = (
            L_data
            + self.w_div * L_div
            + self.w_mom * L_div_vor
            + self.w_mom * L_mom
            + self.w_srt * L_srt
        )
        logs = {
            "loss_total": loss_total,
            "loss_data": L_data,
            "loss_div": L_div,
            "loss_div_vor": L_div_vor,
            "loss_mom": L_mom,
            "loss_srt": L_srt,
            "w_data": tf.constant(1.0),
            "w_div": self.w_div,
            "w_div_vor": self.w_mom,
            "w_mom": self.w_mom,
            "w_srt": tf.constant(1.0),
        }
        return loss_total, logs