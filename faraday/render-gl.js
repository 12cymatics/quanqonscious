'use strict';

/* The surface, drawn on the GPU.
 *
 * WHAT MOVES TO THE GPU, AND WHAT DOES NOT. cymatic.html draws the surface with a
 * per-pixel loop over 381 x 381 = 145 161 pixels, with exp, pow and sqrt in the
 * inner loop of its default view. Measured on the container this was written in:
 * 11.6 to 22.2 ms a frame depending on the view, against 16.7 ms for a whole frame
 * at 60 Hz. That loop is pure shading -- every pixel reads its own inputs and
 * writes its own colour, nothing is accumulated across pixels and nothing feeds
 * back into the simulation -- so it is exactly the work a fragment shader does,
 * and this file does it.
 *
 * The PHYSICS stays where it is, in double precision on the CPU. No GPU on an
 * Intel Mac has a double-precision type -- Metal has no `double`, so WGSL and the
 * WebGL backends on top of it have no f64 -- and the solvers are validated to
 * agreements of parts in ten thousand that single precision cannot carry. Here
 * single precision is right, because the output is an 8-bit pixel: an f32
 * evaluation of these formulas differs from the f64 one by parts in ten million,
 * which moves the stored byte only when the value sits within that distance of a
 * rounding boundary, and then by one level.
 *
 * Nor does the GRAIN simulation move. stepGrains costs more than the shading (28 ms
 * a frame measured, for three passes over 60 000 grains), but it draws from one
 * seeded random sequence in grain order and topples grains out of over-full cells
 * in scan order. A parallel version would be a different algorithm with a different
 * outcome, presented under the old one's name. It stays on the CPU, and its counts
 * are uploaded each frame as a texture.
 *
 * PARITY IS ASSERTED, NOT ASSUMED, at two levels. dns/check-page.mjs draws every view
 * with both renderers on the same state and requires the BYTES to agree to within one
 * level per channel, which is the bound the paragraph above derives. And it requires
 * the values BEFORE the 8-bit rounding to agree to within 2e-2 of a level, a bound
 * derived from the GLSL ES 3.00 precision requirements -- because the byte check alone
 * was shown blind to a wrong coefficient: 168 for 169 in one term moved pixels by up to
 * half a level and passed it. The shader's uRaw mode exists for that second check; the
 * page never sets it.
 *
 * IT REFUSES RATHER THAN FALLING BACK. If WebGL2 is not available, or a texture
 * format it needs is missing, or the shaders do not compile, construction throws
 * with the reason, and the page reports why GPU rendering could not start. It
 * never quietly draws with the CPU under the GPU's name.
 */

const RENDER_GL_VIEW = { sand: 0, height: 1, nodal: 2 };   // anything else: optical (3)

const RENDER_GL_VS = `#version 300 es
void main(){
  /* One triangle covering the viewport, positioned from the vertex index alone, so
     there is no vertex buffer to get wrong. */
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p*2.0 - 1.0, 0.0, 1.0);
}`;

/* A line-for-line transcription of renderSurface's per-pixel body. Where the GLSL
   differs from the JavaScript it is for one of two reasons, each noted:
   pow(x, 2.0) is undefined in GLSL for x < 0, so a square is written as a product;
   and the 8-bit quantisation is done explicitly, so the byte does not depend on how
   a particular GPU rounds a float into a unorm channel. */
const RENDER_GL_FS = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform highp sampler2D uCover, uNoise, uEta, uNx, uNy;
uniform highp usampler2D uBin;
uniform int uView, uGR, uRaw;
uniform float uK, uContrast, uTex, uSlope;
out vec4 outColor;

float tone(float x){ return x <= 0.0 ? 0.0 : 255.0*(1.0 - exp(-x/132.0)); }
/* The byte the CPU path stores: a Uint8ClampedArray rounds to the nearest level and
   clamps. Written out as an exact multiple of 1/255, which every conformant GPU
   converts back to that same integer. */
float level(float v){ return clamp(floor(v + 0.5), 0.0, 255.0)/255.0; }

void main(){
  /* gl_FragCoord counts rows from the BOTTOM; the rasters count them from the top. */
  ivec2 p = ivec2(int(gl_FragCoord.x), uGR - 1 - int(gl_FragCoord.y));
  float cov = texelFetch(uCover, p, 0).r;
  float r0 = 5.0, g0 = 7.0, b0 = 10.0;
  if (cov > 0.0){
    float h = texelFetch(uEta, p, 0).r*uK;
    if (uView == 0){
      uint n = texelFetch(uBin, p, 0).r;
      r0 = 14.0; g0 = 16.0; b0 = 20.0;
      if (n > 0u){
        float t = min(1.0, float(n)/2.0), tt = t*t*(3.0 - 2.0*t);
        r0 += (246.0 - r0)*tt; g0 += (234.0 - g0)*tt; b0 += (208.0 - b0)*tt;
      }
    } else if (uView == 1){
      if (h >= 0.0){ r0 = 8.0 + 70.0*h;   g0 = 12.0 + 190.0*h; b0 = 18.0 + 220.0*h; }
      else         { r0 = 8.0 + 224.0*-h; g0 = 12.0 + 128.0*-h; b0 = 18.0 + 78.0*-h; }
    } else if (uView == 2){
      float t = h/(uContrast*0.9), gg = 1.0/(1.0 + t*t), bb = gg*sqrt(gg);
      r0 = 9.0 + 232.0*bb; g0 = 13.0 + 224.0*bb; b0 = 18.0 + 200.0*bb;
    } else {
      float nx = -texelFetch(uNx, p, 0).r*uSlope*uK, ny = -texelFetch(uNy, p, 0).r*uSlope*uK;
      float inv = 1.0/sqrt(nx*nx + ny*ny + 1.0);
      float Nx = nx*inv, Ny = ny*inv, Nz = inv;
      float tilt = length(vec2(Nx, Ny));
      float s = (tilt - 0.34)/0.105;
      float ring = exp(-(s*s));                    /* Math.pow(s, 2): see above */
      float spec = pow(max(0.0, Nz), 30.0);
      float cau = max(0.0, 1.0 - abs(h)/uContrast);
      r0 = 7.0 + 168.0*ring + 20.0*spec + 30.0*cau;
      g0 = 12.0 + 170.0*ring + 62.0*spec + 54.0*cau;
      b0 = 20.0 + 164.0*ring + 96.0*spec + 72.0*cau;
    }
    if (uTex > 0.18){
      float n = texelFetch(uNoise, p, 0).r*uTex*26.0;
      r0 += n; g0 += n; b0 += n;
    }
  }
  float tr = tone(5.0 + (r0 - 5.0)*cov), tg = tone(7.0 + (g0 - 7.0)*cov),
        tb = tone(10.0 + (b0 - 10.0)*cov);
  /* uRaw is for the parity gate alone: the same values BEFORE they are rounded to a
     byte, written to a float target. The byte comparison cannot see an error smaller
     than one level -- a wrong coefficient that moves every pixel by half a level
     passes it -- and these can. */
  outColor = uRaw == 1 ? vec4(tr, tg, tb, 1.0)
                       : vec4(level(tr), level(tg), level(tb), 1.0);
}`;

/* Whether this browser can run the GPU renderer at all, and if not, why -- asked of
   a throwaway canvas so that a refusal leaves nothing behind. */
function renderGlProbe(){
  if (typeof document === 'undefined')
    return { ok: false, reason: 'there is no document, so there is no canvas to draw on.' };
  const gl = document.createElement('canvas').getContext('webgl2');
  if (!gl) return { ok: false,
    reason: 'this browser offers no WebGL2 context -- the GPU is disabled, absent, or '
          + 'blocked for this page.' };
  const d = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  const lose = gl.getExtension('WEBGL_lose_context');
  if (lose) lose.loseContext();
  return { ok: true, renderer: String(renderer) };
}

class SurfaceGL {
  /* `canvas` must be a canvas nothing else has taken a context on. GR is the raster
     side, 381 in the page. */
  constructor(canvas, GR){
    if (!Number.isInteger(GR) || GR <= 0) throw new RangeError(
      'the GPU raster side must be a positive integer.');
    const gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error(
      'the GPU renderer needs WebGL2, and this browser offers no WebGL2 context on '
      + 'the surface canvas. Draw with the CPU instead; this does not do so for you.');
    this.gl = gl; this.GR = GR; this.lost = false; this.lostReason = null;
    this.contextRestored = false; this.initialized = false;
    this.tex = {}; this.uploaded = new Set();
    this.onContextLost = ev => {
      ev.preventDefault();
      this.markContextLost();
    };
    this.onContextRestored = () => {
      /* Restoration invalidates every old GL object. A user must select GPU again:
         neither this event nor a draw call silently restarts the renderer. */
      this.lost = true; this.initialized = false; this.contextRestored = true;
      this.lostReason = 'the GPU context was restored. Select GPU again to rebuild '
        + 'its resources and upload the current surface, or select CPU.';
    };
    canvas.addEventListener('webglcontextlost', this.onContextLost);
    canvas.addEventListener('webglcontextrestored', this.onContextRestored);
    try { this.reinitialize(); }
    catch (e){
      canvas.removeEventListener('webglcontextlost', this.onContextLost);
      canvas.removeEventListener('webglcontextrestored', this.onContextRestored);
      this.releaseResources();
      throw e;
    }
  }

  markContextLost(){
    this.lost = true; this.initialized = false; this.contextRestored = false;
    this.lostReason = 'the GPU dropped this page\'s WebGL context. Select CPU to '
      + 'continue drawing, or select GPU again after the browser restores the context.';
  }

  get ready(){
    if (this.gl.isContextLost()) this.markContextLost();
    return this.initialized && !this.lost;
  }

  assertReady(){
    if (!this.ready) throw new Error(this.lostReason
      || 'the GPU renderer needs explicit initialization before it can draw.');
  }

  releaseResources(){
    const gl = this.gl;
    /* Objects from a lost context are already invalid; do not submit them to a
       restored context. Objects from an intact context can be deleted normally. */
    if (!this.lost && !gl.isContextLost()){
      for (const { t } of Object.values(this.tex)) gl.deleteTexture(t);
      if (this.fboTex) gl.deleteTexture(this.fboTex);
      if (this.fbo) gl.deleteFramebuffer(this.fbo);
      if (this.vao) gl.deleteVertexArray(this.vao);
      if (this.prog) gl.deleteProgram(this.prog);
    }
    this.tex = {}; this.u = {}; this.uploaded.clear();
    this.fboTex = null; this.fbo = null; this.vao = null; this.prog = null;
    this.initialized = false;
  }

  /* Called by an explicit GPU selection, never by a restoration event or draw.
     The caller must upload the current rasters again after this returns. */
  reinitialize(){
    const gl = this.gl;
    if (gl.isContextLost()){
      this.markContextLost();
      throw new Error(this.lostReason);
    }
    this.releaseResources();
    this.lost = false; this.lostReason = null; this.contextRestored = false;
    try { this.initializeResources(); }
    catch (e){ this.releaseResources(); throw e; }
    if (gl.isContextLost()){
      this.markContextLost();
      throw new Error(this.lostReason);
    }
    this.initialized = true;
    return this;
  }

  initializeResources(){
    const gl = this.gl, GR = this.GR;
    const d = gl.getExtension('WEBGL_debug_renderer_info');
    this.renderer = String(d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL)
                             : gl.getParameter(gl.RENDERER));
    gl.canvas.width = GR; gl.canvas.height = GR;

    const compile = (type, src, what) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)){
        const reason = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error(`the GPU renderer's ${what} shader did not compile on this GPU: `
          + reason);
      }
      return sh;
    };
    const prog = this.prog = gl.createProgram(), shaders = [];
    try {
      shaders.push(compile(gl.VERTEX_SHADER, RENDER_GL_VS, 'vertex'));
      shaders.push(compile(gl.FRAGMENT_SHADER, RENDER_GL_FS, 'fragment'));
      for (const sh of shaders) gl.attachShader(prog, sh);
    } finally { for (const sh of shaders) gl.deleteShader(sh); }
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(
      'the GPU renderer\'s shaders did not link on this GPU: ' + gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    this.u = {};
    for (const n of ['uCover', 'uNoise', 'uEta', 'uNx', 'uNy', 'uBin', 'uView', 'uGR',
                     'uRaw', 'uK', 'uContrast', 'uTex', 'uSlope'])
      this.u[n] = gl.getUniformLocation(prog, n);
    this.vao = gl.createVertexArray();

    /* Rows of 16-bit grain counts are 762 bytes, which is not a multiple of four;
       the default unpack alignment of 4 would read them skewed. */
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.tex = {};
    const units = { uCover: 0, uNoise: 1, uEta: 2, uNx: 3, uNy: 4, uBin: 5 };
    for (const [name, unit] of Object.entries(units)){
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      /* NEAREST and no mipmaps, or the texture is incomplete: R32F is not
         filterable without an extension and an integer texture never is, and an
         incomplete texture reads as zero -- a blank surface rather than an error. */
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (name === 'uBin')
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, GR, GR, 0, gl.RED_INTEGER,
                      gl.UNSIGNED_SHORT, null);
      else
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, GR, GR, 0, gl.RED, gl.FLOAT, null);
      gl.uniform1i(this.u[name], unit);
      this.tex[name] = { t, unit };
    }
    const err = gl.getError();
    if (err !== gl.NO_ERROR) throw new Error(
      `the GPU renderer could not allocate its textures on this GPU (GL error ${err}).`);
  }

  validateRaster(name, data){
    if (!Object.hasOwn(this.tex, name)) throw new RangeError(`unknown GPU raster ${name}.`);
    if (!data || data.length !== this.GR*this.GR) throw new RangeError(
      `${name}: ${data && data.length} values for a ${this.GR} x ${this.GR} raster.`);
    for (let i = 0; i < data.length; i++){
      const v = data[i];
      if (!Number.isFinite(v) || !Number.isFinite(Math.fround(v))) throw new RangeError(
        `${name}[${i}]: ${v} is not a finite float32 raster value.`);
      if (name === 'uBin' && (!Number.isInteger(v) || v < 0 || v > 65535))
        throw new RangeError(`${name}[${i}]: ${v} is not an unsigned 16-bit grain count.`);
    }
  }

  checkError(operation){
    this.assertReady();
    const err = this.gl.getError();
    if (err !== this.gl.NO_ERROR) throw new Error(
      `the GPU renderer could not ${operation} (GL error ${err}).`);
  }

  uploadValidated(name, data){
    const gl = this.gl, { t, unit } = this.tex[name];
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    if (name === 'uBin')
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.GR, this.GR, gl.RED_INTEGER,
                       gl.UNSIGNED_SHORT, data);
    else
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.GR, this.GR, gl.RED, gl.FLOAT, data);
    this.checkError(`upload ${name}`);
    this.uploaded.add(name);
  }

  upload(name, data){
    this.assertReady();
    this.validateRaster(name, data);
    this.uploadValidated(name, data);
  }

  uploadRasters(rasters){
    this.assertReady();
    /* Validate the whole batch before changing any texture. A bad final raster
       must not leave a mixture of the old and new field on the GPU. */
    for (const [name, data] of rasters) this.validateRaster(name, data);
    for (const [name, data] of rasters) this.uploadValidated(name, data);
  }

  /* The rasters that change only when the field is rebuilt, never per frame. */
  uploadStatic(cover, noise){ this.uploadRasters([['uCover', cover], ['uNoise', noise]]); }
  uploadField(eta, nx, ny){ this.uploadRasters([['uEta', eta], ['uNx', nx], ['uNy', ny]]); }
  uploadBins(bin){ this.upload('uBin', bin); }

  validateDraw(o){
    this.assertReady();
    for (const n of ['k', 'contrast', 'tex', 'slope'])
      if (!Number.isFinite(o[n]) || !Number.isFinite(Math.fround(o[n])))
        throw new RangeError(`${n}: ${o[n]} is not a finite float32 GPU uniform.`);
    const v = RENDER_GL_VIEW[o.view], shaderView = v === undefined ? 3 : v;
    if ((shaderView === 2 || shaderView === 3) && !(Math.fround(o.contrast) > 0))
      throw new RangeError('contrast must remain positive in float32 for nodal and optical shading.');
    const needed = ['uCover', 'uNoise', 'uEta', 'uNx', 'uNy'];
    if (shaderView === 0) needed.push('uBin');
    for (const name of needed) if (!this.uploaded.has(name)) throw new Error(
      `${name} must be uploaded after GPU initialization before drawing.`);
    return shaderView;
  }

  draw(o, raw){
    const v = this.validateDraw(o);
    const gl = this.gl, u = this.u;
    gl.viewport(0, 0, this.GR, this.GR);
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.uniform1i(u.uView, v);
    gl.uniform1i(u.uGR, this.GR);
    gl.uniform1i(u.uRaw, raw ? 1 : 0);
    gl.uniform1f(u.uK, o.k);
    gl.uniform1f(u.uContrast, o.contrast);
    gl.uniform1f(u.uTex, o.tex);
    gl.uniform1f(u.uSlope, o.slope);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.checkError('draw the surface');
  }

  /* The shading values BEFORE the 8-bit rounding, three per pixel with row 0 at the
     top, read from a float target the gate draws into. For the parity gate only; the
     page never calls it. Refuses where float render targets are unavailable rather
     than reading something else -- the gate then fails, which is right, because it
     can no longer see an error smaller than one level. */
  readRaw(o){
    this.validateDraw(o);
    const gl = this.gl, GR = this.GR;
    if (!this.fbo){
      if (!gl.getExtension('EXT_color_buffer_float')) throw new Error(
        'this GPU cannot render to a float target (EXT_color_buffer_float), so the '
        + 'values before rounding cannot be read back.');
      this.fboTex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, GR, GR, 0, gl.RGBA, gl.FLOAT, null);
      this.fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTex, 0);
      const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error(
        `the float target for reading values before rounding is incomplete (status ${st}).`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    try {
      this.draw(o, true);
      const raw = new Float32Array(GR*GR*4), out = new Float32Array(GR*GR*4);
      gl.readPixels(0, 0, GR, GR, gl.RGBA, gl.FLOAT, raw);
      this.checkError('read the raw surface');
      for (let i = 0; i < raw.length; i++) if (!Number.isFinite(raw[i]))
        throw new Error(`the GPU raw surface contains a nonfinite value at channel ${i}.`);
      for (let y = 0; y < GR; y++)
        out.set(raw.subarray((GR - 1 - y)*GR*4, (GR - y)*GR*4), y*GR*4);
      return out;
    } finally { gl.bindFramebuffer(gl.FRAMEBUFFER, null); }
  }

  /* The frame just drawn, as RGBA bytes with row 0 at the TOP, which is how the CPU
     path's ImageData stores it. Valid only in the same task as draw(): the drawing
     buffer is not preserved past compositing, and does not need to be for display. */
  readRGBA(){
    this.assertReady();
    const gl = this.gl, GR = this.GR, raw = new Uint8Array(GR*GR*4), out = new Uint8Array(GR*GR*4);
    gl.readPixels(0, 0, GR, GR, gl.RGBA, gl.UNSIGNED_BYTE, raw);
    this.checkError('read the surface');
    for (let y = 0; y < GR; y++)
      out.set(raw.subarray((GR - 1 - y)*GR*4, (GR - y)*GR*4), y*GR*4);
    return out;
  }
}

const FARADAY_RENDER_GL = { SurfaceGL, renderGlProbe, RENDER_GL_VS, RENDER_GL_FS, RENDER_GL_VIEW };
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_RENDER_GL;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_RENDER_GL = FARADAY_RENDER_GL;
