def initialize_particles(particles, cfg, seed=12345):
    """
    Initialize particles using the same logic as _init_two_stream,
    selecting CPU/NumPy or GPU/CuPy based on backend.INIT_DEVICE.
    """
    from .backend import INIT_DEVICE, CUPY, to_xp
    import numpy as np

    # ============================================
    # Select device (CPU or GPU)
    # ============================================
    use_cpu = (INIT_DEVICE == "cpu") or (INIT_DEVICE == "auto" and not CUPY)

    if use_cpu:
        # ------------------- CPU initialization -------------------
        xp_cpu = np
        xp_cpu.random.seed(seed)

        Np = particles.Np
        Lx = cfg.Lx
        v0 = float(cfg.v0)
        vth = float(cfg.vth)

        # === 1. Uniform x distribution with perturbations ===
        i = xp_cpu.arange(Np, dtype=float)
        rnd = xp_cpu.random.random(Np).astype(float)
        x = ((i + rnd) / Np) * Lx

        # === 2. Labels ===
        half = Np // 2
        labels = xp_cpu.concatenate((
            xp_cpu.zeros(half, dtype=np.int8),
            xp_cpu.ones(half, dtype=np.int8)
        ))

        # === 3. v = ±v0 + thermal noise ===
        v = vth * xp_cpu.random.standard_normal((Np, 3)).astype(float)
        v[:, 0] += v0 * (1.0 - 2.0 * labels)

        # === 4. permutation ===
        perm = xp_cpu.random.permutation(Np)

        particles.x     = to_xp(x[perm])
        particles.v     = to_xp(v)
        particles.label = to_xp(labels)
        return

    else:
        # ------------------- GPU initialization -------------------
        import cupy as cp
        cp.random.seed(seed)

        xp = cp

        Np = particles.Np
        Lx = cfg.Lx
        v0 = float(cfg.v0)
        vth = float(cfg.vth)

        # === 1. Uniform x distribution with perturbations ===
        i = xp.arange(Np, dtype=xp.float64)
        rnd = xp.random.random(Np).astype(xp.float64)
        x = ((i + rnd) / Np) * Lx

        # === 2. Labels ===
        half = Np // 2
        labels = xp.concatenate((
            xp.zeros(half, dtype=xp.int8),
            xp.ones(half, dtype=xp.int8)
        ))

        # === 3. v = ±v0 + thermal noise ===
        v = vth * xp.random.standard_normal((Np, 3)).astype(xp.float64)
        v[:, 0] += v0 * (1.0 - 2.0 * labels)

        # === 4. permutation ===
        perm = xp.random.permutation(Np)

        particles.x     = x[perm]
        particles.v     = v
        particles.label = labels
        return
