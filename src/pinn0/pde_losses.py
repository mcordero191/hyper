import tensorflow as tf

# ============ Scheduler ============
class WeightScheduler(tf.Module):
    
    def __init__(self, w_min=1e-2, w_max=1e3, ramp_epochs=5000, name="sched"):
        super().__init__(name=name)
        self.w_min = w_min
        self.w_max = w_max
        self.ramp_epochs = ramp_epochs
        self.epoch = tf.Variable(0, trainable=False, dtype=tf.int32)

    def update(self):
        self.epoch.assign_add(1)

    def value(self):
        alpha = tf.cast(self.epoch, tf.float32) / self.ramp_epochs
        alpha = tf.minimum(alpha, 1.0)
        return self.w_min * tf.pow(self.w_max / self.w_min, alpha)


# ============ Uncertainty Weighting ============
class UncertaintyWeight(tf.Module):
    
    def __init__(self, name=None, ini_sigma=0.0, w_floor=1e-6):
        
        super().__init__(name=name)
        self.log_sigma = tf.Variable(ini_sigma, trainable=True, name=f"log_sigma_{name}")
        self.w_floor = w_floor

    def __call__(self, loss_val):
        w_eff = tf.maximum(tf.exp(-2.0 * self.log_sigma), self.w_floor)
        return w_eff * loss_val + self.log_sigma

    @property
    def sigma(self):
        return tf.exp(self.log_sigma)

    @property
    def weight(self):
        return tf.maximum(tf.exp(-2.0 * self.log_sigma), self.w_floor)


# ============ Augmented Lagrangian ============
class AugmentedLagrangian(tf.Module):
    
    def __init__(self, rho=1.0, tol=1e-4, name="AL"):
        
        super().__init__(name=name)
        self.lambda_div = tf.Variable(0.0, trainable=False)
        self.rho = tf.Variable(rho, trainable=False)
        self.tol = tol

    def penalty(self, loss_div_mse):
        
        return self.lambda_div * loss_div_mse + 0.5 * self.rho * tf.square(loss_div_mse)

    def update(self, loss_div_mse):
        
        c_rms = tf.sqrt(loss_div_mse + 1e-12)
        if c_rms > self.tol:
            self.lambda_div.assign_add(self.rho * tf.stop_gradient(c_rms))
            self.lambda_div.assign(tf.clip_by_value(self.lambda_div, -1e2, 1e2))
            
    
class PINNLoss(tf.Module):
     
    def __init__(self, use_uw=False, use_al=False,
                 min_frac_div=10.0, min_frac_mom=1.0, min_frac_div_vor=0.01,
                 ema_beta=0.1, warmup_epochs=200):
        
        super().__init__()
        self.use_uw = use_uw
        self.use_al = use_al
        self.ema_beta = ema_beta
        self.warmup_epochs = warmup_epochs

        # Training epoch counter
        self.epoch = tf.Variable(0, trainable=False, dtype=tf.int32)

        # Minimum fractions
        self.min_frac_div = min_frac_div
        self.min_frac_mom = min_frac_mom
        self.min_frac_div_vor = min_frac_div_vor

        # Smoothed fractions
        self.frac_div_smooth     = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.frac_mom_smooth     = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.frac_div_vor_smooth = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # Uncertainty weights
        if use_uw:
            self.uw_data    = UncertaintyWeight("data")
            self.uw_div     = UncertaintyWeight("div")
            self.uw_mom     = UncertaintyWeight("mom")
            self.uw_div_vor = UncertaintyWeight("div_vor")

        # Augmented Lagrangian
        if use_al:
            self.AL = AugmentedLagrangian(rho=1.0)

    @property
    def trainables(self):
        if self.use_uw:
            return [
                self.uw_data.log_sigma,
                self.uw_div.log_sigma,
                self.uw_mom.log_sigma,
                self.uw_div_vor.log_sigma,
            ]
        else:
            return []
        
    def update(self):
        """Increase epoch counter (call each iteration)."""
        self.epoch.assign_add(1)

    def __call__(self, loss_data, loss_div, loss_div_vor, loss_mom, loss_srt, w_srt=tf.constant(1e-5)):

        # --- Data ---
        L_data = loss_data if not self.use_uw else self.uw_data(loss_data)

        # --- PDE terms (optionally UW) ---
        L_div     = loss_div if not self.use_uw else self.uw_div(loss_div)
        L_mom     = loss_mom if not self.use_uw else self.uw_mom(loss_mom)
        L_div_vor = loss_div_vor if not self.use_uw else self.uw_div_vor(loss_div_vor)

        # --- Early phase: warmup ---
        if self.epoch < self.warmup_epochs:
            w_div = w_mom = w_div_vor = tf.constant(1e-2, dtype=tf.float32)
        else:
            # Compute total
            # L_total_raw = L_data + L_div + L_mom + L_div_vor + w_srt * loss_srt

            # Fractions
            frac_div_now     = L_div     / (L_data + 1e-12)
            frac_mom_now     = L_mom     / (L_data + 1e-12)
            frac_div_vor_now = L_div_vor / (L_data + 1e-12)

            # EMA update
            self.frac_div_smooth.assign(self.ema_beta * self.frac_div_smooth + (1 - self.ema_beta) * frac_div_now)
            self.frac_mom_smooth.assign(self.ema_beta * self.frac_mom_smooth + (1 - self.ema_beta) * frac_mom_now)
            self.frac_div_vor_smooth.assign(self.ema_beta * self.frac_div_vor_smooth + (1 - self.ema_beta) * frac_div_vor_now)

            # Scaling (very gentle)
            w_div     = self.frac_div_smooth #tf.where(self.frac_div_smooth < self.min_frac_div,  self.min_frac_div / (self.frac_div_smooth + 1e-12), 1.0)
            w_mom     = self.frac_mom_smooth #tf.where(self.frac_mom_smooth < self.min_frac_mom,  self.min_frac_mom / (self.frac_mom_smooth + 1e-12), 1.0)
            w_div_vor = self.frac_div_vor_smooth #tf.where(self.frac_div_vor_smooth < self.min_frac_div_vor,  self.min_frac_div_vor / (self.frac_div_vor_smooth + 1e-12), 1.0)

        # --- PDE Loss ---
        L_pde = w_div * L_div + w_mom * L_mom + w_div_vor * L_div_vor

        # --- AL (optional) ---
        L_AL = 0.0
        if self.use_al:
            L_AL = self.AL.penalty(loss_div)

        # --- Total loss ---
        L_total = L_data + L_pde + L_AL + w_srt * loss_srt

        return L_total, {
            "loss_data": loss_data,
            "loss_mom": loss_mom,
            "loss_div": loss_div,
            "loss_div_vor": loss_div_vor,
            "loss_srt": loss_srt,
            "loss_al": L_AL,
            # "frac_pde_now": frac_pde_now,
            # "frac_pde_smooth": self.frac_pde_smooth,
            # "scale_pde": scale,
            "w_data": tf.constant(1.0),
            "w_pde": w_div,
            "w_div": w_div,
            "w_mom": w_mom,
            "w_div_vor": w_div_vor,
            "w_srt": w_srt,
        }
        
        # return L_total, {
        #     "loss_data": loss_data,
        #     "loss_mom": loss_mom,
        #     "loss_div": loss_div,
        #     "loss_div_vor": loss_div_vor,
        #     "loss_srt": loss_srt,
        #     "loss_al": L_AL,
        #     "w_data": 1.0,
        #     "w_div": w_div,
        #     "w_mom": w_mom,
        #     "w_div_vor": w_div_vor,
        #     "epoch": self.epoch,
        # }
 
class PINNLoss3(tf.Module):
     
    def __init__(self, use_uw=False, use_al=False, min_frac_pde=0.5, ema_beta=0.98):
        super().__init__()
        self.use_uw = use_uw
        self.use_al = use_al
        self.min_frac_pde = min_frac_pde
        self.ema_beta = ema_beta

        # Smoothed PDE fraction (keeps memory)
        self.frac_pde_smooth = tf.Variable(0.5, trainable=False, dtype=tf.float32)

        # Uncertainty weights
        if use_uw:
            self.uw_data    = UncertaintyWeight("data")
            self.uw_div     = UncertaintyWeight("div")
            self.uw_mom     = UncertaintyWeight("mom")
            self.uw_div_vor = UncertaintyWeight("div_vor")

        # Augmented Lagrangian
        if use_al:
            self.AL = AugmentedLagrangian(rho=1.0)

    @property
    def trainables(self):
        if self.use_uw:
            return [
                self.uw_data.log_sigma,
                self.uw_div.log_sigma,
                self.uw_mom.log_sigma,
                self.uw_div_vor.log_sigma,
            ]
        else:
            return []
    
    def update(self):
        return
        
    def __call__(self, loss_data, loss_div, loss_div_vor, loss_mom, loss_srt, w_srt=tf.constant(1e-5)):
        # --- Data term ---
        L_data = loss_data if not self.use_uw else self.uw_data(loss_data)

        # --- PDE terms ---
        L_div     = loss_div if not self.use_uw else self.uw_div(loss_div)
        L_mom     = loss_mom if not self.use_uw else self.uw_mom(loss_mom)
        L_div_vor = loss_div_vor if not self.use_uw else self.uw_div_vor(loss_div_vor)

        L_pde_raw = L_div + L_mom + L_div_vor

        # --- Compute raw fraction ---
        total_raw = L_data + L_pde_raw + w_srt * loss_srt
        frac_pde_now = L_pde_raw / (total_raw + 1e-12)

        # --- Smooth fraction (EMA) ---
        self.frac_pde_smooth.assign(
            self.ema_beta * self.frac_pde_smooth + (1.0 - self.ema_beta) * frac_pde_now
        )

        # --- Adaptive scaling (slow response) ---
        frac_used = self.frac_pde_smooth
        scale = tf.where(frac_used < self.min_frac_pde,
                         self.min_frac_pde / (frac_used + 1e-12),
                         1.0)

        # Apply scale
        L_pde = scale * L_pde_raw

        # --- Augmented Lagrangian (optional) ---
        L_AL = 0.0
        if self.use_al:
            L_AL = self.AL.penalty(loss_div)

        # --- Total loss ---
        L_total = L_data + L_pde + L_AL + w_srt * loss_srt

        return L_total, {
            "loss_data": loss_data,
            "loss_mom": loss_mom,
            "loss_div": loss_div,
            "loss_div_vor": loss_div_vor,
            "loss_srt": loss_srt,
            "loss_al": L_AL,
            "frac_pde_now": frac_pde_now,
            "frac_pde_smooth": self.frac_pde_smooth,
            "scale_pde": scale,
            "w_data": tf.constant(1.0),
            "w_pde": scale,
            "w_div": scale,
            "w_mom": scale,
            "w_div_vor": scale,
            "w_srt": w_srt,
        }