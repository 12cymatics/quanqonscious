/* A view of physical solver data. Camera and display magnification never step or
   alter the fluid/particle state. No network dependencies: this source is inlined
   by the standalone builder. Physical coordinates are metres, with z vertical. */
(function scene3dModule(root) {
  'use strict';
  const DEFAULT_CAMERA = Object.freeze({ yaw: -0.72, elevation: 0.66, distance: 3.45, targetX: 0, targetY: 0 });
  const DEFAULT_OPTIONS = Object.freeze({ heightScale: 1, showSurface: true, showParticles: true,
    showVectors: false, showGrid: true, colorMode: 'water', pointSize: 3,
    vectorScaleSeconds: 1, colorScaleM: null, surfaceOpacity: null });
  const finite = (n, name) => { if (!Number.isFinite(n)) throw new TypeError('3D view: ' + name + ' must be finite.'); return n; };
  function copyArray(a, Type, name, stride) {
    if (a == null) return new Type(0);
    if (!ArrayBuffer.isView(a) && !Array.isArray(a)) throw new TypeError('3D view: ' + name + ' must be an array.');
    if (a.length % stride) throw new RangeError('3D view: malformed ' + name + '.');
    const b = new Type(a.length);
    for (let i = 0; i < a.length; i++) b[i] = finite(a[i], name + '[' + i + ']');
    return b;
  }
  function captureFrame(frame) {
    if (!frame || !(finite(frame.R, 'radius') > 0) || !(finite(frame.h, 'depth') > 0))
      throw new RangeError('3D view: positive radius and depth are required.');
    const f = { R: frame.R, h: frame.h, time: finite(frame.time, 'physical time'),
      generation: frame.generation, steps: frame.steps, initial: !!frame.initial,
      source: frame.source == null ? null : JSON.parse(JSON.stringify(frame.source)), positions: copyArray(frame.positions, Float32Array, 'surface positions', 3),
      indices: copyArray(frame.indices, Uint32Array, 'surface triangles', 3),
      particles: copyArray(frame.particles, Float64Array, 'particle positions', 3),
      particleRadii: frame.particleRadii == null ? null : copyArray(frame.particleRadii, Float64Array, 'particle radii', 1),
      flowPositions: copyArray(frame.flowPositions, Float32Array, 'flow positions', 3),
      flowVectors: copyArray(frame.flowVectors, Float32Array, 'flow vectors', 3),
      etaRange: frame.etaRange ? Array.from(frame.etaRange) : null,
      maxSpeed: frame.maxSpeed == null ? null : finite(frame.maxSpeed, 'maximum speed') };
    for (const key of ['provenance', 'units', 'grid', 'flowSampling', 'displayTessellation', 'maxSpeedScope', 'tracerProvenance', 'grainProvenance'])
      if (frame[key] !== undefined) f[key] = JSON.parse(JSON.stringify(frame[key]));
    const n = f.positions.length / 3;
    if (!n || !f.indices.length) throw new RangeError('3D view: an actual surface mesh is required.');
    for (let i = 0; i < frame.indices.length; i++) {
      if (!Number.isInteger(frame.indices[i]) || frame.indices[i] < 0 || frame.indices[i] >= n)
        throw new RangeError('3D view: surface index lies outside its vertex array.');
    }
    if (f.flowVectors.length !== f.flowPositions.length) throw new RangeError('3D view: velocity and position counts differ.');
    if (f.particleRadii) {
      if (f.particleRadii.length !== f.particles.length / 3) throw new RangeError('3D view: particle radius and position counts differ.');
      for (const radius of f.particleRadii) if (!(radius > 0)) throw new RangeError('3D view: particle radii must be positive metres.');
    }
    for (const [key,stride] of [['grainIds',1],['grainDensity',1],['grainVelocities',3],['grainAngularVelocities',3]]) {
      if(frame[key]==null)continue;
      f[key]=copyArray(frame[key],Float64Array,key,stride);
      if(!f.particleRadii||f[key].length!==f.particles.length/3*stride)throw new RangeError('3D view: incorrect '+key+' count.');
      if(key==='grainIds'&&Array.from(f[key]).some(id=>!Number.isSafeInteger(id)||id<0))throw new RangeError('3D view: grain IDs must be nonnegative safe integers.');
      if(key==='grainDensity'&&Array.from(f[key]).some(d=>!(d>0)))throw new RangeError('3D view: grain density must be positive.');
    }
    for (const [key, count] of [['surfaceDisplacement', n], ['particleDisplacement', f.particles.length / 3], ['flowDisplacement', f.flowPositions.length / 3]]) {
      f[key] = frame[key] == null ? null : copyArray(frame[key], Float32Array, key, 1);
      if (f[key] && f[key].length !== count) throw new RangeError('3D view: incorrect ' + key + ' count.');
    }
    return f;
  }
  function cameraValue(camera) {
    const c = { ...DEFAULT_CAMERA, ...camera };
    for (const key of Object.keys(DEFAULT_CAMERA)) finite(c[key], 'camera ' + key);
    if (!(c.distance > 0) || !(Math.abs(c.elevation) < Math.PI / 2))
      throw new RangeError('3D view: camera requires positive distance and elevation strictly between the poles.');
    return c;
  }
  function optionsValue(options, frame) {
    const o = { ...DEFAULT_OPTIONS, ...options };
    for (const key of ['heightScale', 'pointSize', 'vectorScaleSeconds'])
      if (!(finite(o[key], key) > 0)) throw new RangeError('3D view: ' + key + ' must be positive.');
    if (o.colorScaleM != null && !(finite(o.colorScaleM, 'colour scale') > 0)) throw new RangeError('3D view: colour scale must be positive.');
    if (o.surfaceOpacity != null && !(finite(o.surfaceOpacity, 'surface opacity') >= 0 && o.surfaceOpacity <= 1))
      throw new RangeError('3D view: surface opacity must lie between zero and one.');
    if (!['water', 'height', 'nodal'].includes(o.colorMode)) throw new RangeError('3D view: unknown surface colouring.');
    return o;
  }
  function displayedPositions(positions, displacements, frame, heightScale, surface) {
    const out = new Float32Array(positions.length), R = frame.R, h = frame.h;
    for (let i = 0, v = 0; i < positions.length; i += 3, v++) {
      const d = displacements ? displacements[v] : surface ? positions[i + 2] - h : 0;
      out[i] = positions[i] / R;
      out[i + 1] = (positions[i + 2] + (heightScale - 1) * d - h / 2) / R;
      out[i + 2] = -positions[i + 1] / R;
    }
    for (let i = 0; i < out.length; i++) if (!Number.isFinite(out[i])) throw new RangeError('3D view: display coordinates overflow at the requested scale.');
    return out;
  }
  function displayedParticleRadii(frame) {
    if (frame.particleRadii == null) return null;
    const radii = new Float32Array(frame.particleRadii.length);
    for (let i = 0; i < radii.length; i++) {
      radii[i] = frame.particleRadii[i] / frame.R;
      if (!(radii[i] > 0) || !Number.isFinite(radii[i])) throw new RangeError('3D view: particle display radius is not representable.');
    }
    return radii;
  }
  function normalsFor(p, indices) {
    const n = new Float32Array(p.length);
    for (let t = 0; t < indices.length; t += 3) {
      const a = indices[t] * 3, b = indices[t + 1] * 3, c = indices[t + 2] * 3;
      const ax = p[b] - p[a], ay = p[b + 1] - p[a + 1], az = p[b + 2] - p[a + 2];
      const bx = p[c] - p[a], by = p[c + 1] - p[a + 1], bz = p[c + 2] - p[a + 2];
      const x = ay * bz - az * by, y = az * bx - ax * bz, z = ax * by - ay * bx;
      for (const q of [a, b, c]) { n[q] += x; n[q + 1] += y; n[q + 2] += z; }
    }
    for (let i = 0; i < n.length; i += 3) {
      const d = Math.hypot(n[i], n[i + 1], n[i + 2]);
      if (d > 0) { n[i] /= d; n[i + 1] /= d; n[i + 2] /= d; }
      else n[i + 1] = 1;
    }
    return n;
  }
  function sub(a, b) { return a.map((x, i) => x - b[i]); }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function unit(a) { const n = Math.hypot(...a); return a.map(x => x / n); }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cameraBasis(camera) {
    const c = cameraValue(camera), target = [c.targetX, c.targetY, 0], cp = Math.cos(c.elevation);
    const eye = [target[0] + c.distance * cp * Math.sin(c.yaw),
      target[1] + c.distance * Math.sin(c.elevation), c.distance * cp * Math.cos(c.yaw)];
    const back = unit(sub(eye, target)), right = unit(cross([0, 1, 0], back));
    return { eye, right, up: cross(back, right), forward: back.map(x => -x) };
  }
  /* Exact perspective tangent bounds for a sphere, not a constant-pixel point.
     A sphere crossing the eye plane uses the viewport as a conservative bound;
     the fragment ray/sphere intersection still determines its actual silhouette. */
  function projectedSphereBounds(centre, radius, camera, aspect) {
    if (!(finite(radius, 'sphere radius') > 0)) throw new RangeError('3D view: sphere radius must be positive.');
    const basis = cameraBasis(camera), d = sub(centre, basis.eye), depth = dot(d, basis.forward);
    const q = [dot(d, basis.right), dot(d, basis.up)], scale = 1 / Math.tan(42 * Math.PI / 360);
    if (!(depth > radius)) return { min: [-1, -1], max: [1, 1] };
    const denominator = depth * depth - radius * radius;
    const bounds = q.map((x, i) => {
      const tangent = radius * Math.sqrt(denominator + x * x), f = i ? scale : scale / aspect;
      return [(x * depth - tangent) * f / denominator, (x * depth + tangent) * f / denominator];
    });
    return { min: [bounds[0][0], bounds[1][0]], max: [bounds[0][1], bounds[1][1]] };
  }
  function multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
      for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
    return out;
  }
  function matrixFor(camera, aspect) {
    const c = cameraValue(camera), target = [c.targetX, c.targetY, 0];
    const cp = Math.cos(c.elevation), eye = [target[0] + c.distance * cp * Math.sin(c.yaw),
      target[1] + c.distance * Math.sin(c.elevation), c.distance * cp * Math.cos(c.yaw)];
    const z = unit(sub(eye, target)), x = unit(cross([0, 1, 0], z)), y = cross(z, x);
    const view = new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
    const f = 1 / Math.tan(42 * Math.PI / 360), near = 0.01, far = Math.max(100, c.distance * 4);
    const projection = new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0,
      0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0]);
    return multiply(projection, view);
  }
  function projectPoint(point, camera, aspect) {
    const m = matrixFor(camera, aspect), p = [...point, 1], q = new Array(4).fill(0);
    for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) q[r] += m[k * 4 + r] * p[k];
    return { x: q[0] / q[3], y: q[1] / q[3], depth: q[2] / q[3], w: q[3] };
  }
  const VERTEX = `#version 300 es
precision highp float;
in vec3 aPosition;
in vec3 aNormal;
in float aValue;
in vec3 aCenter;
in float aRadius;
uniform mat4 uMatrix;
uniform float uPointSize;
uniform int uInstanced;
uniform vec3 uEye;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uForward;
uniform vec2 uProjectionScale;
out vec3 vNormal;
out float vValue;
flat out vec3 vCenter;
flat out float vRadius;
out vec3 vRay;
void main(){
 vNormal=aNormal;vValue=aValue;vCenter=aCenter;vRadius=aRadius;vRay=vec3(0.0);
 if(uInstanced==1){
   vec3 d=aCenter-uEye;float depth=dot(d,uForward);
   vec2 c=vec2(dot(d,uRight),dot(d,uUp));vec2 lo=vec2(-1.0),hi=vec2(1.0);
   float den=depth*depth-aRadius*aRadius;
   if(depth>aRadius&&den>0.0){
     vec2 tangent=aRadius*sqrt(vec2(den)+c*c);
     lo=(c*depth-tangent)*uProjectionScale/den;hi=(c*depth+tangent)*uProjectionScale/den;
   }
   vec2 p=mix(lo,hi,aPosition.xy*0.5+0.5);
   gl_Position=vec4(p,0.0,1.0);
   vRay=uForward+uRight*(p.x/uProjectionScale.x)+uUp*(p.y/uProjectionScale.y);
 }else gl_Position=uMatrix*vec4(aPosition,1.0);
 gl_PointSize=uPointSize;
}`;
  const FRAGMENT = `#version 300 es
precision highp float;
in vec3 vNormal;
in float vValue;
flat in vec3 vCenter;
flat in float vRadius;
in vec3 vRay;
uniform vec3 uEye;
uniform mat4 uMatrix;
uniform vec4 uColor;
uniform int uKind;
uniform int uMode;
uniform float uColorScale;
out vec4 fragColor;
void main(){
 vec3 col=uColor.rgb;float alpha=uColor.a;
#ifdef FINITE_GRAINS
 gl_FragDepth=gl_FragCoord.z;
#endif
 if(uKind==1){
   vec2 d=gl_PointCoord*2.0-1.0;float r2=dot(d,d);if(r2>1.0)discard;
   float shade=0.6+0.4*sqrt(1.0-r2);col*=shade;
 }
#ifdef FINITE_GRAINS
 else if(uKind==3){
   vec3 ray=normalize(vRay),toCenter=vCenter-uEye;
   float along=dot(toCenter,ray);
   vec3 perpendicular=toCenter-along*ray;
   float disc=vRadius*vRadius-dot(perpendicular,perpendicular);
   if(disc<0.0)discard;
   float root=sqrt(disc),t=along-root;
   vec4 clip=uMatrix*vec4(uEye+t*ray,1.0);float depth=clip.z/clip.w;
   if(t<=0.0||clip.w<=0.0||depth< -1.0){
     t=along+root;clip=uMatrix*vec4(uEye+t*ray,1.0);depth=clip.z/clip.w;
   }
   if(t<=0.0||clip.w<=0.0||depth< -1.0||depth>1.0)discard;
   gl_FragDepth=0.5+0.5*depth;
   vec3 n=normalize((t-along)*ray-perpendicular);
   float diffuse=max(dot(n,normalize(vec3(-0.5,0.9,0.35))),0.0);
   col*=0.45+0.55*diffuse;
 }
#endif
 else if(uKind==2){
   vec3 n=normalize(vNormal);if(!gl_FrontFacing)n=-n;
   float diffuse=abs(dot(n,normalize(vec3(-0.5,0.9,0.35))));
   float light=0.44+0.56*diffuse;
   if(uMode==1){float t=0.5+0.5*tanh(vValue/uColorScale);col=mix(vec3(0.15,0.32,0.55),vec3(0.32,0.84,0.88),t);}
   if(uMode==2){float band=exp(-pow(vValue/uColorScale,2.0)*12.0);col=mix(vec3(0.05,0.12,0.17),vec3(0.80,0.83,0.72),band);}
   col*=light;
 }
 fragColor=vec4(col,alpha);
}`;
  function renderExportWorker() {
    self.onmessage = event => {
      const { size, snapshot } = event.data;
      let renderer = null;
      try {
        if (typeof OffscreenCanvas !== 'function') {
          self.postMessage({ unsupported: true, reason: 'OffscreenCanvas is unavailable in this browser worker.' });
          return;
        }
        const output = new OffscreenCanvas(size, size);
        const gpu = output.getContext('webgl2', { alpha: false, antialias: true, depth: true,
          preserveDrawingBuffer: true, premultipliedAlpha: false });
        if (!gpu) {
          self.postMessage({ unsupported: true, reason: 'Offscreen WebGL 2 is unavailable in this browser worker.' });
          return;
        }
        const limit = Math.min(gpu.getParameter(gpu.MAX_RENDERBUFFER_SIZE), gpu.getParameter(gpu.MAX_TEXTURE_SIZE));
        if (size > limit) throw new RangeError('The export worker cannot allocate the requested ' + size + '-pixel image. Resolution was not reduced.');
        renderer = FARADAY_SCENE3D.create(output, { interactive: false, camera: snapshot.camera, options: snapshot.options });
        renderer.setFrame(snapshot.frame); renderer.resize(size, size, 1); renderer.render();
        const rgba = new Uint8Array(size * size * 4);
        gpu.readPixels(0, 0, size, size, gpu.RGBA, gpu.UNSIGNED_BYTE, rgba);
        if (gpu.getError() !== gpu.NO_ERROR || gpu.isContextLost()) throw new Error('The export worker could not read the complete image.');
        self.postMessage({ ok: true, rgba: rgba.buffer, samples: gpu.getParameter(gpu.SAMPLES) }, [rgba.buffer]);
      } catch (error) {
        self.postMessage({ ok: false, error: String(error && error.message || error) });
      } finally {
        if (renderer) renderer.dispose(true);
      }
    };
  }
  function exportWorkerSource() {
    // Both functions are already part of the bundled module. Their source carries
    // the exact same renderer and shaders into an offline Blob worker.
    return '(' + scene3dModule.toString() + ')(globalThis);(' + renderExportWorker.toString() + ')();';
  }
  function renderInExportWorker(size, snapshot) {
    if (typeof root.Worker !== 'function' || typeof root.OffscreenCanvas !== 'function')
      return Promise.resolve({ unsupported: true, reason: 'This browser does not expose worker-backed canvas rendering.' });
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(new Blob([exportWorkerSource()], { type: 'text/javascript' }));
      let worker, timer;
      const cleanup = () => { if (timer) clearTimeout(timer); if (worker) worker.terminate(); URL.revokeObjectURL(url); };
      try { worker = new Worker(url); }
      catch (error) { cleanup(); resolve({ unsupported: true, reason: 'The browser refused the export worker: ' + String(error.message || error) }); return; }
      timer = setTimeout(() => { cleanup(); reject(new Error('The full-resolution export worker did not finish within 60 seconds. No reduced image was saved.')); }, 60000);
      worker.onerror = event => { cleanup(); reject(new Error('The 3D export worker failed: ' + (event.message || 'no error details'))); };
      worker.onmessageerror = () => { cleanup(); reject(new Error('The 3D export worker returned an unreadable image.')); };
      worker.onmessage = event => {
        const result = event.data; cleanup();
        if (result.unsupported) { resolve(result); return; }
        if (!result.ok) { reject(new Error(result.error || 'The 3D export worker failed.')); return; }
        if (!(result.rgba instanceof ArrayBuffer) || result.rgba.byteLength !== size * size * 4) {
          reject(new Error('The 3D export worker returned an incomplete image.')); return;
        }
        resolve(result);
      };
      try { worker.postMessage({ size, snapshot }); }
      catch (error) { cleanup(); reject(new Error('The captured 3D export could not be sent to its worker: ' + String(error.message || error))); }
    });
  }
  function create(canvas, callbacks) {
    if (!canvas || typeof canvas.getContext !== 'function') throw new TypeError('3D view requires a canvas.');
    callbacks = callbacks || {};
    let frame = null, options = optionsValue(callbacks.options), camera = cameraValue(callbacks.camera);
    let gl = null, program = null, grainProgram = null, uniforms, grainUniforms, buffers = [], layers = null, lost = false, disposed = false;
    let revision = 0, renderCount = 0, drawCalls = 0, lastWidth = 0, lastHeight = 0, pixelRatio = 1, drag = null, resizeObserver = null;
    const report = error => { if (callbacks.onError) callbacks.onError(error); };
    function shader(type, source) {
      const sh = gl.createShader(type); gl.shaderSource(sh, source); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) { const error = gl.getShaderInfoLog(sh); gl.deleteShader(sh); throw new Error('3D shader: ' + error); }
      return sh;
    }
    function init() {
      gl = canvas.getContext('webgl2', { alpha: false, antialias: true, depth: true,
        preserveDrawingBuffer: true, premultipliedAlpha: false });
      if (!gl) throw new Error('The interactive 3D view requires WebGL 2. The fluid solver has not been replaced.');
      function link(fragment) {
        const vs = shader(gl.VERTEX_SHADER, VERTEX), fs = shader(gl.FRAGMENT_SHADER, fragment);
        const p = gl.createProgram(); gl.attachShader(p, vs); gl.attachShader(p, fs); gl.linkProgram(p);
        gl.deleteShader(vs); gl.deleteShader(fs);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('3D shader link: ' + gl.getProgramInfoLog(p));
        const u = {};
        for (const name of ['uMatrix', 'uPointSize', 'uColor', 'uKind', 'uMode', 'uColorScale', 'uInstanced', 'uEye', 'uRight', 'uUp', 'uForward', 'uProjectionScale']) u[name] = gl.getUniformLocation(p, name);
        return { program: p, uniforms: u };
      }
      const standard = link(FRAGMENT);
      program = standard.program; uniforms = standard.uniforms;
      // A separate compiled variant confines explicit sphere depth to grains.
      // Writing gl_FragDepth even unchanged changes MSAA on the existing layers.
      const grains = link(FRAGMENT.replace('#version 300 es\n', '#version 300 es\n#define FINITE_GRAINS\n'));
      grainProgram = grains.program; grainUniforms = grains.uniforms;
      lost = false;
      if (frame) rebuild();
    }
    function clearBuffers() { if (gl && !lost) for (const b of buffers) gl.deleteBuffer(b); buffers = []; }
    function buffer(data, target) {
      const b = gl.createBuffer(); buffers.push(b); gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW); return b;
    }
    function layer(p, indices, normals, values, mode) {
      return { p: buffer(p, gl.ARRAY_BUFFER), n: buffer(normals || new Float32Array(p.length), gl.ARRAY_BUFFER),
        v: buffer(values || new Float32Array(p.length / 3), gl.ARRAY_BUFFER),
        indices: indices ? buffer(indices, gl.ELEMENT_ARRAY_BUFFER) : null,
        count: indices ? indices.length : p.length / 3, mode };
    }
    function rebuild() {
      if (!frame || lost || disposed) return;
      clearBuffers();
      const p = displayedPositions(frame.positions, frame.surfaceDisplacement, frame, options.heightScale, true);
      const eta = new Float32Array(p.length / 3);
      for (let i = 0; i < eta.length; i++) eta[i] = frame.surfaceDisplacement ? frame.surfaceDisplacement[i] : frame.positions[i * 3 + 2] - frame.h;
      const surface = layer(p, frame.indices, normalsFor(p, frame.indices), eta, gl.TRIANGLES);
      const centres = displayedPositions(frame.particles, frame.particleDisplacement, frame, options.heightScale, false);
      let particles;
      if (frame.particleRadii) {
        // Two triangles bound each exact ray-traced sphere. Geometry and fragment
        // depth come from the supplied metre-valued radius, not the quad itself.
        particles = layer(new Float32Array([-1,-1,0, 1,-1,0, 1,1,0, -1,1,0]), new Uint32Array([0,1,2,0,2,3]), null, null, gl.TRIANGLES);
        particles.instances = frame.particleRadii.length;
        particles.centres = buffer(centres, gl.ARRAY_BUFFER);
        particles.radii = buffer(displayedParticleRadii(frame), gl.ARRAY_BUFFER);
      } else particles = layer(centres, null, null, null, gl.POINTS);
      const level = frame.h / (2 * frame.R), floor = [], floorIndices = [], rim = [], grid = [];
      const segments = 128;
      floor.push(0, -level, 0);
      for (let i = 0; i <= segments; i++) {
        const a = i * Math.PI * 2 / segments, x = Math.cos(a), z = Math.sin(a);
        floor.push(x, -level, z);
        if (i < segments) floorIndices.push(0, i + 1, i + 2);
        if (i < segments) {
          const b = (i + 1) * Math.PI * 2 / segments, x2 = Math.cos(b), z2 = Math.sin(b);
          rim.push(x, -level, z, x2, -level, z2, x, level, z, x2, level, z2);
          if (!(i % 16)) rim.push(x, -level, z, x, level, z);
        }
      }
      for (let i = -4; i <= 4; i++) {
        const q = i / 5, edge = Math.sqrt(1 - q * q);
        grid.push(-edge, -level + 0.0001, q, edge, -level + 0.0001, q,
          q, -level + 0.0001, -edge, q, -level + 0.0001, edge);
      }
      const flow = [], fp = displayedPositions(frame.flowPositions, frame.flowDisplacement, frame, options.heightScale, false);
      for (let i = 0; i < fp.length; i += 3) {
        const vx = frame.flowVectors[i] * options.vectorScaleSeconds / frame.R;
        const vy = frame.flowVectors[i + 2] * options.vectorScaleSeconds / frame.R;
        const vz = -frame.flowVectors[i + 1] * options.vectorScaleSeconds / frame.R;
        if (vx === 0 && vy === 0 && vz === 0) continue;
        const start = [fp[i], fp[i + 1], fp[i + 2]], end = [start[0] + vx, start[1] + vy, start[2] + vz];
        flow.push(...start, ...end);
        const length = Math.hypot(vx, vy, vz), axis = [vx / length, vy / length, vz / length];
        const side = unit(cross(axis, Math.abs(axis[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]));
        for (const sign of [-1, 1]) flow.push(...end, end[0] - vx * 0.2 + side[0] * length * 0.09 * sign,
          end[1] - vy * 0.2 + side[1] * length * 0.09 * sign,
          end[2] - vz * 0.2 + side[2] * length * 0.09 * sign);
      }
      layers = { surface, particles,
        floor: layer(new Float32Array(floor), new Uint32Array(floorIndices), null, null, gl.TRIANGLES),
        rim: layer(new Float32Array(rim), null, null, null, gl.LINES),
        grid: layer(new Float32Array(grid), null, null, null, gl.LINES),
        flow: layer(new Float32Array(flow), null, null, null, gl.LINES) };
      revision++;
    }
    function bind(l, active) {
      for (const [name, b, count] of [['aPosition', l.p, 3], ['aNormal', l.n, 3], ['aValue', l.v, 1]]) {
        const a = gl.getAttribLocation(active, name); gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, count, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(a, 0);
      }
      for (const [name, b, count] of [['aCenter', l.centres, 3], ['aRadius', l.radii, 1]]) {
        const a = gl.getAttribLocation(active, name);
        if (b) {
          gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.enableVertexAttribArray(a);
          gl.vertexAttribPointer(a, count, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(a, 1);
        } else {
          gl.disableVertexAttribArray(a); gl.vertexAttribDivisor(a, 0); gl.vertexAttrib4f(a, 0, 0, 0, 1);
        }
      }
      if (l.indices) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, l.indices);
    }
    function draw(l, color, kind, depthWrite) {
      if (!l.count || l.instances === 0) return;
      const active = kind === 3 ? grainProgram : program, u = kind === 3 ? grainUniforms : uniforms;
      gl.useProgram(active);
      drawCalls++; bind(l, active); gl.uniform4fv(u.uColor, color); gl.uniform1i(u.uKind, kind);
      gl.uniform1i(u.uInstanced, l.instances == null ? 0 : 1);
      gl.depthMask(depthWrite);
      if (l.instances != null) gl.drawElementsInstanced(l.mode, l.count, gl.UNSIGNED_INT, 0, l.instances);
      else if (l.indices) gl.drawElements(l.mode, l.count, gl.UNSIGNED_INT, 0); else gl.drawArrays(l.mode, 0, l.count);
    }
    function resize(width, height, ratio) {
      if (disposed) return false;
      width = width || canvas.clientWidth || canvas.width; height = height || canvas.clientHeight || canvas.height;
      ratio = ratio || (callbacks.interactive === false ? 1 : root.devicePixelRatio || 1);
      const w = Math.round(width * ratio), h = Math.round(height * ratio);
      if (!(w > 0 && h > 0)) return false;
      if (w !== canvas.width || h !== canvas.height) { canvas.width = w; canvas.height = h; }
      lastWidth = w; lastHeight = h; pixelRatio = ratio; return true;
    }
    function render() {
      if (disposed || lost || !frame || !layers) return false;
      if (!lastWidth) resize();
      drawCalls = 0; gl.viewport(0, 0, canvas.width, canvas.height); gl.clearColor(0.0275, 0.0392, 0.0549, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      const basis = cameraBasis(camera), scale = 1 / Math.tan(42 * Math.PI / 360);
      for (const [active, u] of [[program, uniforms], [grainProgram, grainUniforms]]) {
        gl.useProgram(active); gl.uniformMatrix4fv(u.uMatrix, false, matrixFor(camera, canvas.width / canvas.height));
        for (const [name, value] of [['uEye', basis.eye], ['uRight', basis.right], ['uUp', basis.up], ['uForward', basis.forward]])
          gl.uniform3fv(u[name], value);
        gl.uniform2f(u.uProjectionScale, scale * canvas.height / canvas.width, scale);
        gl.uniform1f(u.uPointSize, options.pointSize * (callbacks.interactive === false ? canvas.width / 800 : pixelRatio));
        gl.uniform1i(u.uMode, options.colorMode === 'height' ? 1 : options.colorMode === 'nodal' ? 2 : 0);
        gl.uniform1f(u.uColorScale, options.colorScaleM || frame.h);
      }
      draw(layers.floor, [0.045, 0.071, 0.094, 1], 0, true);
      if (options.showGrid) draw(layers.grid, [0.14, 0.20, 0.25, 0.5], 0, false);
      if (options.showVectors) draw(layers.flow, [0.24, 0.79, 0.91, 0.95], 0, true);
      if (options.showParticles) draw(layers.particles, [0.91, 0.77, 0.49, 1], frame.particleRadii ? 3 : 1, true);
      const surfaceOpacity = options.surfaceOpacity == null ? options.showParticles ? 0.42 : 0.82 : options.surfaceOpacity;
      if (options.showSurface && surfaceOpacity > 0) draw(layers.surface, [0.12, 0.52, 0.61, surfaceOpacity], 2, surfaceOpacity === 1);
      draw(layers.rim, [0.43, 0.60, 0.67, 0.75], 0, false);
      gl.depthMask(true); gl.flush(); renderCount++; return true;
    }
    function changeCamera(next) {
      camera = cameraValue(next); render(); if (callbacks.onCameraChange) callbacks.onCameraChange({ ...camera });
    }
    function pointerDown(event) {
      if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
      drag = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button !== 0 };
      if (canvas.setPointerCapture) canvas.setPointerCapture(event.pointerId); event.preventDefault();
    }
    function pointerMove(event) {
      if (!drag) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y; drag.x = event.clientX; drag.y = event.clientY;
      if (drag.pan) changeCamera({ ...camera, targetX: camera.targetX - dx * camera.distance / 1000, targetY: camera.targetY + dy * camera.distance / 1000 });
      else {
        const elevation = Math.max(-1.54, Math.min(1.54, camera.elevation + dy * 0.006));
        changeCamera({ ...camera, yaw: camera.yaw - dx * 0.006, elevation });
      }
      event.preventDefault();
    }
    function pointerUp() { drag = null; }
    function wheel(event) {
      const distance = camera.distance * Math.exp(event.deltaY * 0.001);
      if (distance > 0.15 && distance < 60) changeCamera({ ...camera, distance });
      event.preventDefault();
    }
    function key(event) {
      const c = { ...camera }; let handled = true;
      if (event.key === 'ArrowLeft') c.yaw -= 0.1;
      else if (event.key === 'ArrowRight') c.yaw += 0.1;
      else if (event.key === 'ArrowUp') c.elevation = Math.min(1.54, c.elevation + 0.1);
      else if (event.key === 'ArrowDown') c.elevation = Math.max(-1.54, c.elevation - 0.1);
      else if (event.key === '+' || event.key === '=') c.distance *= 0.9;
      else if (event.key === '-') c.distance *= 1.1;
      else if (event.key.toLowerCase() === 'r') Object.assign(c, DEFAULT_CAMERA);
      else if (event.key.toLowerCase() === 't') { c.yaw = 0; c.elevation = Math.PI / 2 - 0.001; }
      else handled = false;
      if (handled) { changeCamera(c); event.preventDefault(); }
    }
    function contextLost(event) {
      event.preventDefault(); lost = true;
      report(new Error('3D graphics context lost. Simulation state is retained; waiting for graphics restoration.'));
      if (callbacks.onContextChange) callbacks.onContextChange({ lost: true });
    }
    function contextRestored() {
      if (disposed) return;
      try { buffers = []; init(); render(); if (callbacks.onContextChange) callbacks.onContextChange({ lost: false }); }
      catch (error) { report(error); }
    }
    const preventMenu = event => event.preventDefault();
    const listeners = [];
    const listen = (type, fn, opts) => { canvas.addEventListener(type, fn, opts); listeners.push([type, fn, opts]); };
    listen('webglcontextlost', contextLost); listen('webglcontextrestored', contextRestored);
    if (callbacks.interactive !== false) {
      canvas.tabIndex = canvas.tabIndex >= 0 ? canvas.tabIndex : 0; canvas.style.touchAction = 'none';
      canvas.setAttribute('aria-label', 'Three-dimensional fluid cell. Drag to orbit, Shift-drag to pan, wheel to zoom. Arrow keys orbit; R resets; T views from above.');
      listen('pointerdown', pointerDown); listen('pointermove', pointerMove); listen('pointerup', pointerUp);
      listen('pointercancel', pointerUp); listen('lostpointercapture', pointerUp); listen('wheel', wheel, { passive: false });
      listen('keydown', key); listen('contextmenu', preventMenu);
    }
    if (callbacks.interactive !== false && typeof root.ResizeObserver === 'function') {
      resizeObserver = new root.ResizeObserver(() => { resize(); render(); }); resizeObserver.observe(canvas);
    }
    try { init(); } catch (error) { for (const [t, fn, opts] of listeners) canvas.removeEventListener(t, fn, opts); if (resizeObserver) resizeObserver.disconnect(); report(error); throw error; }
    const api = {
      setFrame(next) { frame = captureFrame(next); rebuild(); return api; },
      setOptions(next) { options = optionsValue({ ...options, ...next }, frame); if (frame) rebuild(); return api; },
      render, resize,
      camera: () => ({ ...camera }),
      setCamera: next => { changeCamera({ ...camera, ...next }); return api; },
      resetCamera: () => { changeCamera(DEFAULT_CAMERA); return api; },
      topView: () => { changeCamera({ ...DEFAULT_CAMERA, yaw: 0, elevation: Math.PI / 2 - 0.001 }); return api; },
      options: () => ({ ...options }),
      snapshot: () => { if (!frame) throw new Error('No solved or specified initial frame is available.'); return { frame: captureFrame(frame), camera: { ...camera }, options: { ...options },
          pointSizePerPixel: options.pointSize * pixelRatio / canvas.width, width: canvas.width, height: canvas.height }; },
      async exportCanvas(size, captured, capturedCamera) {
        if (!Number.isInteger(size) || size <= 0) throw new RangeError('Export size must be a positive integer.');
        if (lost) throw new Error('Cannot export while the graphics context is lost.');
        const supplied = captured && captured.frame ? captured : { frame: captured || frame, camera: capturedCamera || camera, options,
          pointSizePerPixel: options.pointSize * pixelRatio / canvas.width };
        if (!supplied.frame) throw new Error('No solved or specified initial frame is available.');
        // Capture everything before the first yield: the live solver, camera or
        // user inputs may change while the requested image is being rendered.
        const snap = { frame: captureFrame(supplied.frame), camera: cameraValue(supplied.camera),
          options: optionsValue(supplied.options), pointSizePerPixel: supplied.pointSizePerPixel };
        const limit = Math.min(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE), gl.getParameter(gl.MAX_TEXTURE_SIZE));
        const expectedSamples = gl.getParameter(gl.SAMPLES);
        if (size > limit) throw new RangeError('Requested image exceeds this GPU limit of ' + limit + ' pixels. Resolution was not reduced.');
        const yieldUI = () => new Promise(resolve => setTimeout(resolve, 0));
        const exportOptions = { ...snap.options };
        if (snap.pointSizePerPixel != null) exportOptions.pointSize = snap.pointSizePerPixel * 800;
        const workerImage = await renderInExportWorker(size, { ...snap, options: exportOptions });
        if (!workerImage.unsupported && workerImage.samples !== expectedSamples) {
          workerImage.unsupported = true;
          workerImage.reason = 'Offscreen rendering supplied ' + workerImage.samples
            + ' antialias samples instead of the displayed scene\'s ' + expectedSamples + '.';
          workerImage.rgba = null;
        }
        if (!workerImage.unsupported) {
          const image = canvas.ownerDocument.createElement('canvas'); image.width = image.height = size;
          const ctx = image.getContext('2d');
          if (!ctx) throw new Error('Could not allocate the export image canvas.');
          const rgba = new Uint8Array(workerImage.rgba), stripRows = 128, rowBytes = size * 4;
          const flipped = new Uint8ClampedArray(rowBytes * Math.min(stripRows, size));
          for (let y = 0; y < size; y += stripRows) {
            await yieldUI();
            const rows = Math.min(stripRows, size - y), bytes = rowBytes * rows;
            for (let row = 0; row < rows; row++)
              flipped.set(rgba.subarray((y + row) * rowBytes, (y + row + 1) * rowBytes), (rows - row - 1) * rowBytes);
            ctx.putImageData(new ImageData(flipped.subarray(0, bytes), size, rows), 0, size - y - rows);
          }
          image.faradayExportMode = 'worker-webgl2';
          return image;
        }
        if (callbacks.onExportStatus) callbacks.onExportStatus('Rendering the same full-resolution 3D image on the main thread: ' + workerImage.reason);
        await yieldUI();
        // Initialize the independent context with a tiny drawing buffer. On
        // software-backed browsers, creating a new context and allocating its
        // full 4K attachments together is a long synchronous operation. The
        // requested resolution is allocated below, after the next UI yield.
        const output = canvas.ownerDocument.createElement('canvas'); output.width = output.height = 1;
        const renderer = create(output, { interactive: false, camera: snap.camera, options: exportOptions });
        try {
          await yieldUI(); renderer.setFrame(snap.frame); renderer.resize(size, size, 1);
          await yieldUI(); renderer.render();
          const gpu = output.getContext('webgl2');
          if (gpu.getParameter(gpu.SAMPLES) !== expectedSamples)
            throw new Error('The export context cannot preserve the displayed antialias quality. No reduced-quality image was saved.');
          // A full-image drawImage would synchronously wait for all GPU work and
          // readback. A fence permits the main UI to run while that work finishes.
          const fence = gpu.fenceSync(gpu.SYNC_GPU_COMMANDS_COMPLETE, 0);
          if (!fence) throw new Error('Could not create the image-render completion fence.');
          gpu.flush();
          const deadline = performance.now() + 60000;
          try {
            for (;;) {
              const status = gpu.clientWaitSync(fence, 0, 0);
              if (status === gpu.ALREADY_SIGNALED || status === gpu.CONDITION_SATISFIED) break;
              if (status === gpu.WAIT_FAILED || gpu.isContextLost()) throw new Error('The export graphics context failed before its image completed.');
              if (performance.now() > deadline) throw new Error('The requested image did not finish on the GPU within 60 seconds. No reduced image was saved.');
              await yieldUI();
            }
          } finally { gpu.deleteSync(fence); }
          const image = canvas.ownerDocument.createElement('canvas'); image.width = image.height = size;
          const ctx = image.getContext('2d');
          if (!ctx) throw new Error('Could not allocate the export image canvas.');
          // Read the completed image in strips. This is an exact row-order
          // conversion from WebGL bottom-up RGBA to Canvas top-down RGBA; it
          // never rerenders, resamples or reduces the requested resolution.
          const stripRows = 128, rowBytes = size * 4;
          const raw = new Uint8Array(rowBytes * Math.min(stripRows, size));
          const flipped = new Uint8ClampedArray(raw.length);
          for (let y = 0; y < size; y += stripRows) {
            await yieldUI();
            const rows = Math.min(stripRows, size - y), bytes = rowBytes * rows;
            gpu.readPixels(0, y, size, rows, gpu.RGBA, gpu.UNSIGNED_BYTE, raw.subarray(0, bytes));
            if (gpu.getError() !== gpu.NO_ERROR) throw new Error('GPU image readback failed; no partial image was saved.');
            for (let row = 0; row < rows; row++)
              flipped.set(raw.subarray(row * rowBytes, (row + 1) * rowBytes), (rows - row - 1) * rowBytes);
            ctx.putImageData(new ImageData(flipped.subarray(0, bytes), size, rows), 0, size - y - rows);
          }
          image.faradayExportMode = 'main-thread-webgl2';
          return image;
        } finally { renderer.dispose(true); }
      },
      stats: () => ({ ready: !!frame && !lost && !disposed, lost, revision, renderCount, drawCalls,
        renderer: 'WebGL2 depth-tested 3D', surfaceVertices: frame ? frame.positions.length / 3 : 0,
        triangles: frame ? frame.indices.length / 3 : 0, particles: frame ? frame.particles.length / 3 : 0,
        particleGeometry: frame && frame.particleRadii ? 'physical-radius spheres' : 'tracer points',
        time: frame ? frame.time : null, generation: frame ? frame.generation : null,
        steps: frame ? frame.steps : null, initial: frame ? frame.initial : null,
        heightScale: options.heightScale, camera: { ...camera }, width: canvas.width, height: canvas.height }),
      dispose(releaseContext) { if (disposed) return; if (resizeObserver) resizeObserver.disconnect(); for (const [t, fn, opts] of listeners) canvas.removeEventListener(t, fn, opts);
        clearBuffers(); if (!lost) { if (program) gl.deleteProgram(program); if (grainProgram) gl.deleteProgram(grainProgram); } disposed = true;
        if (releaseContext && !lost) { const extension = gl.getExtension('WEBGL_lose_context'); if (extension) extension.loseContext(); } }
    };
    return api;
  }
  const api = Object.freeze({ create, captureFrame, displayedPositions, displayedParticleRadii, projectedSphereBounds, normalsFor, matrixFor, projectPoint, exportWorkerSource, DEFAULT_CAMERA, DEFAULT_OPTIONS });
  root.FARADAY_SCENE3D = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
