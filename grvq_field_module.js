export class GRVQFieldModule {
  constructor({
    radialResolution = 64,
    thetaResolution = 48,
    phiResolution = 96,
    rMin = 0.0,
    rMax = 6.0,
    turyavrttiFactor = 0.75,
    sutraCoefficients = null,
  } = {}) {
    this.radialResolution = radialResolution;
    this.thetaResolution = thetaResolution;
    this.phiResolution = phiResolution;
    this.rMin = rMin;
    this.rMax = rMax;
    this.turyavrttiFactor = turyavrttiFactor;
    this.setSutraCoefficients(sutraCoefficients);
    this.chebyshevGrid = this.#buildChebyshevGrid();
    this.shells = this.#initializeShells();
  }

  setSutraCoefficients(coefficients) {
    if (!coefficients) {
      this.sutraCoefficients = null;
      return;
    }
    if (!Array.isArray(coefficients) || coefficients.length !== 29) {
      throw new Error("Sutra coefficients must be an array of 29 values.");
    }
    this.sutraCoefficients = coefficients.map((value) => Number(value));
  }

  #buildChebyshevGrid() {
    const n = this.radialResolution;
    const grid = new Float64Array(n);
    const mid = 0.5 * (this.rMax + this.rMin);
    const half = 0.5 * (this.rMax - this.rMin);
    for (let i = 0; i < n; i += 1) {
      const angle = (Math.PI * i) / (n - 1);
      grid[i] = mid + half * Math.cos(angle);
    }
    return grid;
  }

  #initializeShells() {
    const shells = [];
    for (let rIndex = 0; rIndex < this.radialResolution; rIndex += 1) {
      const shell = new Float64Array(this.thetaResolution * this.phiResolution);
      shells.push(shell);
    }
    return shells;
  }

  #index(thetaIndex, phiIndex) {
    return thetaIndex * this.phiResolution + phiIndex;
  }

  computeR4Suppression(r) {
    const r2 = r * r;
    const core = 1.0 + r2;
    return 1.0 / (1.0 + r2 * r2 / core);
  }

  recursiveLog(value, depth = 4) {
    let accumulator = value;
    for (let i = 0; i < depth; i += 1) {
      const scale = 1.0 + (i + 1) * 0.25;
      const term = Math.log1p(Math.abs(accumulator) * scale);
      accumulator = Math.sign(accumulator) * (term + accumulator / (1.0 + scale));
    }
    return accumulator;
  }

  #sutraBlend(theta, phi) {
    if (!this.sutraCoefficients) {
      return 0.0;
    }
    let accumulator = 0.0;
    const length = this.sutraCoefficients.length;
    for (let i = 0; i < length; i += 1) {
      const weight = this.sutraCoefficients[i];
      const harmonic =
        Math.sin((i + 1) * theta) * Math.cos((i + 1) * phi) +
        Math.cos((i + 1) * theta) * Math.sin((i + 1) * phi);
      accumulator += weight * harmonic;
    }
    return this.recursiveLog(accumulator, 4);
  }

  #vedicStabilizer(r, theta, phi) {
    const s1 = Math.sin(theta) * Math.cos(phi);
    const s2 = Math.cos(theta) * Math.sin(phi);
    const s3 = Math.sin(r + theta + phi);
    const s4 = Math.cos(2.0 * (r + theta + phi));
    const s5 = Math.sin(3.0 * r) * Math.cos(theta - phi);
    const composite = s1 * s2 + 0.5 * s3 + 0.25 * s4 + 0.15 * s5;
    return this.recursiveLog(composite, 5);
  }

  #turyavrttiModulation(r, theta, phi) {
    const phase = Math.PI * r * theta * phi;
    return 1.0 + this.turyavrttiFactor * Math.sin(phase);
  }

  updateShell(rIndex, timestep = 0) {
    const r = this.chebyshevGrid[rIndex];
    const shell = this.shells[rIndex];
    const suppression = this.computeR4Suppression(r);

    for (let tIndex = 0; tIndex < this.thetaResolution; tIndex += 1) {
      const theta = (Math.PI * tIndex) / Math.max(1, this.thetaResolution - 1);
      for (let pIndex = 0; pIndex < this.phiResolution; pIndex += 1) {
        const phi = (2.0 * Math.PI * pIndex) / Math.max(1, this.phiResolution);
        const stabilizer = this.#vedicStabilizer(r, theta, phi);
        const modulation = this.#turyavrttiModulation(r, theta, phi);
        const sutraBlend = this.#sutraBlend(theta, phi);
        const recursive = this.recursiveLog(stabilizer * modulation, 3);
        const value = suppression * recursive * Math.exp(-0.05 * r * r);
        const phase = 1.0 + 0.25 * Math.sin(0.15 * timestep + theta + phi);
        shell[this.#index(tIndex, pIndex)] = value * phase * (1.0 + 0.2 * sutraBlend);
      }
    }
    return shell;
  }

  step(timestep = 0) {
    const output = [];
    for (let rIndex = 0; rIndex < this.radialResolution; rIndex += 1) {
      output.push(this.updateShell(rIndex, timestep));
    }
    return output;
  }

  exportShells() {
    return this.shells.map((shell, idx) => ({
      r: this.chebyshevGrid[idx],
      data: Array.from(shell),
      thetaResolution: this.thetaResolution,
      phiResolution: this.phiResolution,
    }));
  }
}

if (typeof window !== "undefined") {
  window.GRVQFieldModule = GRVQFieldModule;
}
