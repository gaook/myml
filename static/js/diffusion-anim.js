/* Canvas animation web-components for the Diffusion Appendix deck.
   All elements are self-animating; pause globally with
   document.documentElement.dataset.animPaused = '1'. */
(function () {
  'use strict';

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) {
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  function theme() {
    return {
      accent: cssVar('--accent', '#54D6E8'),
      accent2: cssVar('--accent2', '#E8A554'),
      ink: cssVar('--ink', '#E8ECF4'),
      muted: cssVar('--muted', '#8B95A8'),
      line: cssVar('--line', '#232B3B'),
      panel: cssVar('--panel', '#141A26')
    };
  }

  class AnimBase extends HTMLElement {
    connectedCallback() {
      if (this._started) return;
      this._started = true;
      this.style.display = 'block';
      this.canvas = document.createElement('canvas');
      this.canvas.style.cssText = 'width:100%;height:100%;display:block;';
      this.appendChild(this.canvas);
      this.ctx = this.canvas.getContext('2d');
      this.t = 0;
      const resize = () => {
        const w = this.clientWidth || 800, h = this.clientHeight || 460;
        if (w === this.W && h === this.H) return;
        this.W = w; this.H = h;
        this.canvas.width = w * 2; this.canvas.height = h * 2;
        this.ctx.setTransform(2, 0, 0, 2, 0, 0);
      };
      resize();
      try { new ResizeObserver(resize).observe(this); } catch (e) {}
      if (this.init) this.init();
      const loop = () => {
        if (!this.isConnected) return;
        resize();
        const paused = document.documentElement.dataset.animPaused === '1';
        if (!paused) { this.t += 1 / 60; if (this.step) this.step(1 / 60); }
        if (this.W > 4) this.draw(theme());
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
    clear(bg) {
      this.ctx.clearRect(0, 0, this.W, this.H);
      if (bg) { this.ctx.fillStyle = bg; this.ctx.fillRect(0, 0, this.W, this.H); }
    }
    arrow(x, y, dx, dy, color, lw) {
      const c = this.ctx, L = Math.hypot(dx, dy);
      if (L < 0.5) return;
      c.strokeStyle = color; c.fillStyle = color; c.lineWidth = lw || 1.6;
      c.beginPath(); c.moveTo(x, y); c.lineTo(x + dx, y + dy); c.stroke();
      const a = Math.atan2(dy, dx), s = Math.min(7, 3 + L * 0.18);
      c.beginPath();
      c.moveTo(x + dx, y + dy);
      c.lineTo(x + dx - s * Math.cos(a - 0.45), y + dy - s * Math.sin(a - 0.45));
      c.lineTo(x + dx - s * Math.cos(a + 0.45), y + dy - s * Math.sin(a + 0.45));
      c.closePath(); c.fill();
    }
    label(txt, x, y, color, size, align) {
      const c = this.ctx;
      c.fillStyle = color; c.font = (size || 15) + 'px "IBM Plex Mono", monospace';
      c.textAlign = align || 'left'; c.fillText(txt, x, y); c.textAlign = 'left';
    }
  }

  /* ── ODE vector field with flowing trajectories (limit-cycle field) ── */
  class AnimOdeField extends AnimBase {
    field(x, y) { // normalized coords centered at 0, radius ~1
      const r2 = x * x + y * y;
      return { vx: -y + x * (1 - r2) * 0.9, vy: x + y * (1 - r2) * 0.9 };
    }
    init() {
      const rnd = mulberry32(7);
      this.pts = Array.from({ length: 14 }, () => ({
        x: (rnd() - 0.5) * 2.4, y: (rnd() - 0.5) * 2.4, trail: []
      }));
    }
    step(dt) {
      for (const p of this.pts) {
        const v = this.field(p.x, p.y);
        p.x += v.vx * dt * 0.9; p.y += v.vy * dt * 0.9;
        p.trail.push([p.x, p.y]);
        if (p.trail.length > 90) p.trail.shift();
      }
    }
    draw(th) {
      this.clear();
      const S = Math.min(this.W, this.H) / 3.0, cx = this.W / 2, cy = this.H / 2;
      const X = (x) => cx + x * S, Y = (y) => cy - y * S;
      // arrow grid
      const n = 11;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const x = -1.45 + 2.9 * i / (n - 1), y = -1.45 + 2.9 * j / (n - 1);
        const v = this.field(x, y), L = Math.hypot(v.vx, v.vy) + 1e-6;
        const k = Math.min(0.13, L * 0.1) / L;
        this.arrow(X(x), Y(y), v.vx * k * S, -v.vy * k * S, th.line, 1.4);
      }
      for (let pi = 0; pi < this.pts.length; pi++) {
        const p = this.pts[pi], col = pi % 2 ? th.accent : th.accent2;
        const c = this.ctx;
        c.strokeStyle = col; c.lineWidth = 2; c.globalAlpha = 0.75;
        c.beginPath();
        p.trail.forEach(([x, y], k) => { k ? c.lineTo(X(x), Y(y)) : c.moveTo(X(x), Y(y)); });
        c.stroke(); c.globalAlpha = 1;
        c.fillStyle = col;
        c.beginPath(); c.arc(X(p.x), Y(p.y), 4.5, 0, 7); c.fill();
      }
      this.label('v(x, t)', 14, 24, th.muted);
      this.label('trajectories x(t)', 14, 46, th.accent);
    }
  }

  /* ── Brownian motion sample paths + √t envelope ── */
  class AnimBrownian extends AnimBase {
    init() { this.reset(); }
    reset() {
      this.T = 0;
      this.rnd = mulberry32((Math.random() * 1e9) | 0);
      this.paths = Array.from({ length: 7 }, () => [0]);
    }
    step(dt) {
      this.T += dt * 0.22;
      if (this.T > 1.06) { this.reset(); return; }
      const sdt = Math.sqrt(dt * 0.22);
      for (const p of this.paths) p.push(p[p.length - 1] + sdt * gauss(this.rnd));
    }
    draw(th) {
      this.clear();
      const padL = 46, padR = 16, padY = 26;
      const X = (f) => padL + f * (this.W - padL - padR);
      const mid = this.H / 2, sc = (this.H / 2 - padY) / 1.6;
      const c = this.ctx;
      // axis
      c.strokeStyle = th.line; c.lineWidth = 1;
      c.beginPath(); c.moveTo(padL, mid); c.lineTo(this.W - padR, mid); c.stroke();
      // envelope ±2√t
      c.setLineDash([5, 5]); c.strokeStyle = th.accent2; c.lineWidth = 1.6;
      for (const s of [1, -1]) {
        c.beginPath();
        for (let i = 0; i <= 100; i++) {
          const f = i / 100, y = mid - s * 2 * Math.sqrt(f) * sc * 0.5;
          i ? c.lineTo(X(f), y) : c.moveTo(X(f), y);
        }
        c.stroke();
      }
      c.setLineDash([]);
      const N = this.paths[0].length, steps = Math.max(2, Math.ceil(1.06 / (0.22 / 60)));
      this.paths.forEach((p, pi) => {
        c.strokeStyle = pi === 0 ? th.accent : th.accent;
        c.globalAlpha = pi === 0 ? 1 : 0.38;
        c.lineWidth = pi === 0 ? 2.4 : 1.5;
        c.beginPath();
        for (let i = 0; i < N; i++) {
          const f = i / steps;
          const y = mid - p[i] * sc * 0.5;
          i ? c.lineTo(X(f), y) : c.moveTo(X(f), y);
        }
        c.stroke();
      });
      c.globalAlpha = 1;
      this.label('w(t)', 10, mid - sc * 0.9, th.muted);
      this.label('t', this.W - 22, mid + 20, th.muted);
      this.label('±2√t', X(0.86), mid - 2 * Math.sqrt(0.9) * sc * 0.5 - 10, th.accent2);
    }
  }

  /* ── Euler steps vs exact trajectory ── */
  class AnimEuler extends AnimBase {
    f(x, t) { return Math.cos(3.2 * t) * 2.2 - 0.9 * x; }
    init() {
      // exact (fine RK4)
      this.exact = [];
      let x = -1.1;
      const h = 0.002;
      for (let t = 0; t <= 4.0001; t += h) {
        this.exact.push([t, x]);
        const k1 = this.f(x, t), k2 = this.f(x + h / 2 * k1, t + h / 2),
          k3 = this.f(x + h / 2 * k2, t + h / 2), k4 = this.f(x + h * k3, t + h);
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
      }
      this.h = 0.45;
      this.euler = [[0, -1.1]];
      this.heun = [[0, -1.1]];
      this.phase = 0;
    }
    step() {
      this.phase += 1;
      if (this.phase % 26 !== 0) return;
      const h = this.h;
      let [t, x] = this.euler[this.euler.length - 1];
      if (t < 4 - 1e-9) this.euler.push([t + h, x + h * this.f(x, t)]);
      let [t2, y] = this.heun[this.heun.length - 1];
      if (t2 < 4 - 1e-9) {
        const p = y + h * this.f(y, t2);
        this.heun.push([t2 + h, y + h / 2 * (this.f(y, t2) + this.f(p, t2 + h))]);
      }
      if (t >= 4 - 1e-9 && t2 >= 4 - 1e-9 && this.phase > 60 * 26) {
        this.euler = [[0, -1.1]]; this.heun = [[0, -1.1]]; this.phase = 0;
      }
    }
    draw(th) {
      this.clear();
      const padL = 40, padR = 14;
      const X = (t) => padL + t / 4 * (this.W - padL - padR);
      const mid = this.H * 0.55, sc = this.H / 3.4;
      const Y = (x) => mid - x * sc * 0.8;
      const c = this.ctx;
      c.strokeStyle = th.line; c.beginPath(); c.moveTo(padL, mid); c.lineTo(this.W - padR, mid); c.stroke();
      // exact
      c.strokeStyle = th.ink; c.lineWidth = 2; c.globalAlpha = 0.85; c.beginPath();
      this.exact.forEach(([t, x], i) => { i ? c.lineTo(X(t), Y(x)) : c.moveTo(X(t), Y(x)); });
      c.stroke(); c.globalAlpha = 1;
      const poly = (pts, col) => {
        c.strokeStyle = col; c.fillStyle = col; c.lineWidth = 2;
        c.beginPath();
        pts.forEach(([t, x], i) => { i ? c.lineTo(X(t), Y(x)) : c.moveTo(X(t), Y(x)); });
        c.stroke();
        pts.forEach(([t, x]) => { c.beginPath(); c.arc(X(t), Y(x), 4, 0, 7); c.fill(); });
      };
      poly(this.euler, th.accent2);
      poly(this.heun, th.accent);
      this.label('exact x(t)', padL + 6, 22, th.ink);
      this.label('Euler (h=0.45)', padL + 6, 44, th.accent2);
      this.label('Heun (h=0.45)', padL + 6, 66, th.accent);
    }
  }

  /* ── 1D density transport: mixture → Gaussian.  mode="ode" | "sde" ── */
  class AnimDensity extends AnimBase {
    init() {
      this.mode = this.getAttribute('mode') || 'ode';
      this.N = 320;
      this.resetPts();
      this.phase = 0; // 0..1 progress, then hold, then reset
      this.hold = 0;
    }
    resetPts() {
      const rnd = mulberry32(11);
      // bimodal start
      this.x0 = Array.from({ length: this.N }, () =>
        (rnd() < 0.55 ? -1.3 : 1.5) + gauss(rnd) * 0.34);
      // matched Gaussian targets (sorted pairing → monotone map, no crossing)
      const z = Array.from({ length: this.N }, () => gauss(rnd) * 0.75);
      const xi = this.x0.map((v, i) => i).sort((a, b) => this.x0[a] - this.x0[b]);
      const zs = z.slice().sort((a, b) => a - b);
      this.z = new Array(this.N);
      xi.forEach((idx, rank) => { this.z[idx] = zs[rank]; });
      this.x = this.x0.slice();
      this.jit = this.x0.map((_, i) => (i * 0.37) % 1);
      this.rnd = mulberry32(999);
    }
    step(dt) {
      if (this.phase >= 1) {
        this.hold += dt;
        if (this.hold > 1.4) { this.phase = 0; this.hold = 0; this.resetPts(); }
        return;
      }
      const sp = 0.24;
      this.phase = Math.min(1, this.phase + dt * sp);
      const s = this.phase;
      if (this.mode === 'sde') {
        // OU-like: drift toward matched target + noise that fades near the end
        const g = 0.9 * (1 - s * 0.85);
        for (let i = 0; i < this.N; i++) {
          this.x[i] += (this.z[i] - this.x[i]) * dt * sp / Math.max(1 - s + 0.06, 0.06)
            + g * Math.sqrt(dt * sp) * gauss(this.rnd) * 1.35;
        }
      } else {
        for (let i = 0; i < this.N; i++) this.x[i] = (1 - s) * this.x0[i] + s * this.z[i];
      }
    }
    density(xs, xmin, xmax, bins) {
      const h = new Array(bins).fill(0);
      for (const v of xs) {
        const b = Math.floor((v - xmin) / (xmax - xmin) * bins);
        if (b >= 0 && b < bins) h[b]++;
      }
      // smooth
      const s = h.map((_, i) => (h[i - 2] || 0) * 0.1 + (h[i - 1] || 0) * 0.22 + h[i] * 0.36 + (h[i + 1] || 0) * 0.22 + (h[i + 2] || 0) * 0.1);
      return s;
    }
    draw(th) {
      this.clear();
      const xmin = -3.4, xmax = 3.4, bins = 72;
      const X = (v) => (v - xmin) / (xmax - xmin) * this.W;
      const base = this.H * 0.86;
      const c = this.ctx;
      c.strokeStyle = th.line; c.beginPath(); c.moveTo(0, base); c.lineTo(this.W, base); c.stroke();
      // density curve
      const d = this.density(this.x, xmin, xmax, bins);
      const mx = Math.max(...d, 1);
      c.beginPath();
      c.moveTo(0, base);
      d.forEach((v, i) => {
        c.lineTo((i + 0.5) / bins * this.W, base - v / mx * (this.H * 0.62));
      });
      c.lineTo(this.W, base);
      c.closePath();
      c.fillStyle = th.accent + '2A'; c.fill();
      c.strokeStyle = th.accent; c.lineWidth = 2.2; c.stroke();
      // particles
      c.fillStyle = th.accent2;
      for (let i = 0; i < this.N; i += 2) {
        const y = base + 6 + this.jit[i] * (this.H * 0.09);
        c.globalAlpha = 0.8;
        c.beginPath(); c.arc(X(this.x[i]), y, 2.2, 0, 7); c.fill();
      }
      c.globalAlpha = 1;
      this.label('p_t(x)', 12, 24, th.accent);
      this.label('particles x(t)', 12, 46, th.accent2);
      this.label(this.mode === 'sde' ? 'SDE: drift + noise' : 'ODE: deterministic flow',
        this.W - 12, 24, th.muted, 15, 'right');
      this.label('t = ' + this.phase.toFixed(2), this.W - 12, 46, th.ink, 15, 'right');
    }
  }

  /* ── Itô demo: paths of dx = σ dw with running E[x²] vs σ²t ── */
  class AnimIto extends AnimBase {
    init() { this.reset(); }
    reset() {
      this.T = 0;
      this.rnd = mulberry32((Math.random() * 1e9) | 0);
      this.M = 240;
      this.xs = new Array(this.M).fill(0);
      this.shown = Array.from({ length: 6 }, () => [0]);
      this.m2 = [0];
    }
    step(dt) {
      this.T += dt * 0.2;
      if (this.T > 1.05) { this.reset(); return; }
      const sdt = Math.sqrt(dt * 0.2);
      for (let i = 0; i < this.M; i++) this.xs[i] += sdt * gauss(this.rnd);
      for (let i = 0; i < 6; i++) this.shown[i].push(this.xs[i]);
      this.m2.push(this.xs.reduce((a, v) => a + v * v, 0) / this.M);
    }
    draw(th) {
      this.clear();
      const padL = 44, padR = 14;
      const X = (f) => padL + f * (this.W - padL - padR);
      const mid = this.H * 0.40, sc = this.H * 0.16;
      const c = this.ctx;
      c.strokeStyle = th.line; c.beginPath(); c.moveTo(padL, mid); c.lineTo(this.W - padR, mid); c.stroke();
      const steps = Math.ceil(1.05 / (0.2 / 60));
      // sample paths (top)
      c.globalAlpha = 0.4; c.lineWidth = 1.4; c.strokeStyle = th.accent;
      for (const p of this.shown) {
        c.beginPath();
        p.forEach((v, i) => { const y = mid - v * sc; i ? c.lineTo(X(i / steps), y) : c.moveTo(X(i / steps), y); });
        c.stroke();
      }
      c.globalAlpha = 1;
      // bottom: E[x²] vs t line
      const b0 = this.H * 0.94, bh = this.H * 0.30;
      c.strokeStyle = th.line; c.beginPath(); c.moveTo(padL, b0); c.lineTo(this.W - padR, b0); c.stroke();
      c.setLineDash([5, 5]); c.strokeStyle = th.muted; c.lineWidth = 1.6;
      c.beginPath(); c.moveTo(X(0), b0); c.lineTo(X(1), b0 - bh); c.stroke(); c.setLineDash([]);
      c.strokeStyle = th.accent2; c.lineWidth = 2.4; c.beginPath();
      this.m2.forEach((v, i) => { const y = b0 - v / 1.05 * bh; i ? c.lineTo(X(i / steps), y) : c.moveTo(X(i / steps), y); });
      c.stroke();
      this.label('sample paths of dx = σ dw', padL + 4, 20, th.accent);
      this.label('E[x²] (ensemble avg)', padL + 4, this.H * 0.60, th.accent2);
      this.label('σ²t', X(0.9), b0 - bh * 0.98, th.muted);
    }
  }

  /* ── Control-volume flux: particles flow, box counts density ── */
  class AnimFlux extends AnimBase {
    field(x, y) { // gentle swirl toward a moving sink
      const sx = Math.cos(this.t * 0.5) * 0.5, sy = Math.sin(this.t * 0.35) * 0.3;
      return { vx: -(x - sx) * 0.45 - (y - sy) * 0.5, vy: -(y - sy) * 0.45 + (x - sx) * 0.5 };
    }
    init() {
      const rnd = mulberry32(21);
      this.rnd = rnd;
      this.pts = Array.from({ length: 240 }, () => ({ x: (rnd() - 0.5) * 3, y: (rnd() - 0.5) * 2 }));
    }
    step(dt) {
      for (const p of this.pts) {
        const v = this.field(p.x, p.y);
        p.x += v.vx * dt; p.y += v.vy * dt;
        p.x += gauss(this.rnd) * 0.012; p.y += gauss(this.rnd) * 0.012;
      }
    }
    draw(th) {
      this.clear();
      const S = Math.min(this.W / 3.2, this.H / 2.2), cx = this.W / 2, cy = this.H / 2;
      const X = (x) => cx + x * S, Y = (y) => cy - y * S;
      const c = this.ctx;
      // box [-0.45,0.45]^2
      const inBox = this.pts.filter(p => Math.abs(p.x) < 0.45 && Math.abs(p.y) < 0.45).length;
      const tint = Math.min(0.55, inBox / 110);
      c.fillStyle = th.accent; c.globalAlpha = tint * 0.5;
      c.fillRect(X(-0.45), Y(0.45), 0.9 * S, 0.9 * S);
      c.globalAlpha = 1;
      c.strokeStyle = th.accent; c.lineWidth = 2;
      c.strokeRect(X(-0.45), Y(0.45), 0.9 * S, 0.9 * S);
      // flux arrows at face midpoints
      const faces = [[-0.45, 0, -1, 0], [0.45, 0, 1, 0], [0, 0.45, 0, 1], [0, -0.45, 0, -1]];
      for (const [fx, fy, nx, ny] of faces) {
        const v = this.field(fx, fy);
        const flux = v.vx * nx + v.vy * ny; // outward positive
        const col = flux > 0 ? th.accent2 : th.accent;
        this.arrow(X(fx), Y(fy), nx * flux * 90 || (flux * 90) * nx, -ny * flux * 90, col, 2.4);
      }
      c.fillStyle = th.ink;
      for (const p of this.pts) {
        c.globalAlpha = 0.55;
        c.beginPath(); c.arc(X(p.x), Y(p.y), 2, 0, 7); c.fill();
      }
      c.globalAlpha = 1;
      this.label('control volume V', X(-0.45), Y(0.45) - 10, th.accent);
      this.label('inside: ' + inBox + ' particles', 14, 24, th.ink);
      this.label('outflux', 14, 46, th.accent2);
      this.label('influx', 14, 68, th.accent);
    }
  }

  /* ── Girsanov: one path, two drift hypotheses ── */
  class AnimGirsanov extends AnimBase {
    init() {
      const rnd = mulberry32(4242);
      // one fixed wiggly path
      this.path = [];
      let x = -0.05;
      const n = 260;
      for (let i = 0; i <= n; i++) {
        const t = i / n;
        this.path.push([t, x]);
        x += (0.9 * Math.sin(t * 5)) * (1 / n) + Math.sqrt(1 / n) * gauss(rnd) * 0.55;
      }
      this.k = 0;
      this.logs = [0, 0];
    }
    driftF(x, t) { return 0.9 * Math.sin(t * 5); }        // hypothesis f (close to truth)
    driftG(x, t) { return -1.3 * x; }                      // hypothesis f~
    step() {
      this.k += 1.4;
      if (this.k >= this.path.length - 1) { this.k = 0; this.logs = [0, 0]; return; }
      const i = Math.floor(this.k);
      const [t, x] = this.path[i];
      const dx = this.path[i + 1][1] - x, dt = 1 / (this.path.length - 1);
      const ll = (mu) => -Math.pow(dx - mu * dt, 2) / (2 * 0.55 * 0.55 * dt);
      this.logs[0] += ll(this.driftF(x, t)) * 0.02;
      this.logs[1] += ll(this.driftG(x, t)) * 0.02;
    }
    draw(th) {
      this.clear();
      const padL = 30, padR = 170;
      const X = (t) => padL + t * (this.W - padL - padR);
      const mid = this.H * 0.5, sc = this.H * 0.30;
      const Y = (x) => mid - x * sc;
      const c = this.ctx;
      c.strokeStyle = th.line; c.beginPath(); c.moveTo(padL, mid); c.lineTo(this.W - padR, mid); c.stroke();
      // the one observed path
      c.strokeStyle = th.ink; c.lineWidth = 2.4; c.beginPath();
      this.path.forEach(([t, x], i) => { i ? c.lineTo(X(t), Y(x)) : c.moveTo(X(t), Y(x)); });
      c.stroke();
      // drift arrows along path
      for (let i = 12; i < this.path.length; i += 26) {
        const [t, x] = this.path[i];
        this.arrow(X(t), Y(x), 26, -this.driftF(x, t) * 22, th.accent, 2);
        this.arrow(X(t), Y(x), 26, -this.driftG(x, t) * 22, th.accent2, 2);
      }
      // moving marker
      const i = Math.floor(this.k), [t, x] = this.path[Math.min(i, this.path.length - 1)];
      c.fillStyle = th.ink; c.beginPath(); c.arc(X(t), Y(x), 6, 0, 7); c.fill();
      // likelihood bars
      const bx = this.W - padR + 30, bw = 34, bb = this.H * 0.82, bh = this.H * 0.55;
      const v0 = Math.exp(this.logs[0]), v1 = Math.exp(this.logs[1]);
      const m = Math.max(v0, v1, 1e-6);
      c.fillStyle = th.accent; c.fillRect(bx, bb - bh * (v0 / m), bw, bh * (v0 / m));
      c.fillStyle = th.accent2; c.fillRect(bx + bw + 18, bb - bh * (v1 / m), bw, bh * (v1 / m));
      this.label('p_f', bx + 4, bb + 20, th.accent);
      this.label('p_f̃', bx + bw + 22, bb + 20, th.accent2);
      this.label('same path x_{0:T}', padL + 4, 24, th.ink);
      this.label('drift f', padL + 4, 46, th.accent);
      this.label('drift f̃', padL + 4, 68, th.accent2);
      this.label('relative likelihood', bx - 6, bb + 44, th.muted, 13);
    }
  }

  customElements.define('anim-ode-field', AnimOdeField);
  customElements.define('anim-brownian', AnimBrownian);
  customElements.define('anim-euler', AnimEuler);
  customElements.define('anim-density', AnimDensity);
  customElements.define('anim-ito', AnimIto);
  customElements.define('anim-flux', AnimFlux);
  customElements.define('anim-girsanov', AnimGirsanov);
})();
