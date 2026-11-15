'use strict';

/* Passive material markers in the computed liquid. They are not finite sand
 * grains: no settling, grain contact, inertia, random kicks, packing, or force
 * feedback is inferred. x and y are metres; sigma=z/H is dimensionless.
 *
 * Each fluid step supplies its complete old and new observations. Their
 * mapped-coordinate velocity fields are interpolated linearly in physical
 * time, then integrated with the embedded Cash–Karp Runge–Kutta 5(4) pair.
 * Failed error estimates or stage boundary crossings bisect that time
 * interval. Coordinates are never projected, clipped, reflected, or repaired.
 * A failed step leaves every particle and its clock unchanged.
 */
const FARADAY_CELL3D_TRACERS = (() => {
  const TAU=2*Math.PI, GOLDEN_ANGLE=Math.PI*(3-Math.sqrt(5));
  const DEFAULT_SEED=0x46524144;
  const A=[0,1/5,3/10,3/5,1,7/8];
  const B=[[],[1/5],[3/40,9/40],[3/10,-9/10,6/5],
    [-11/54,5/2,-70/27,35/27],
    [1631/55296,175/512,575/13824,44275/110592,253/4096]];
  const C5=[37/378,0,250/621,125/594,0,512/1771];
  const C4=[2825/27648,0,18575/48384,13525/55296,277/14336,1/4];
  class BoundaryCrossing extends RangeError {}
  function finite(v,name){
    if(typeof v!=='number'||!Number.isFinite(v))throw new TypeError(`${name} must be finite.`);
    return v;
  }
  function positive(v,name){finite(v,name);if(!(v>0))throw new RangeError(`${name} must be positive.`);return v;}
  function seedValue(v){
    if(!Number.isInteger(v)||v<0||v>0xffffffff)throw new RangeError('Tracer seed must be a uint32 integer.');
    return v;
  }
  function observerOf(value,options={}){
    if(value && (typeof value.heightAt==='function'||typeof value.surfaceAt==='function'))return value;
    const OBS=typeof FARADAY_CELL3D_OBSERVATION!=='undefined'?FARADAY_CELL3D_OBSERVATION:
      typeof require==='function'?require('./cell3d-observation.js'):globalThis.FARADAY_CELL3D_OBSERVATION;
    if(!OBS)throw new Error('The nonlinear observation module is required for flow tracers.');
    return OBS.createObserver(value,options);
  }
  function validateObserver(o,needsVelocity=false){
    positive(o.R,'Observation radius');finite(o.time,'Observation physical time');
    if(o.time<0)throw new RangeError('Observation time cannot be negative.');
    if(!Number.isInteger(o.generation)||o.generation<0)throw new RangeError('Observation generation must be nonnegative.');
    if(typeof o.heightAt!=='function'&&typeof o.surfaceAt!=='function')
      throw new TypeError('A liquid height observation is required.');
    if(needsVelocity&&(typeof o.surfaceAt!=='function'||typeof o.velocityAt!=='function'))
      throw new TypeError('A complete nonlinear velocity and surface observation is required.');
  }
  function mixedSeed(seed){
    let x=seed>>>0;x^=x>>>16;x=Math.imul(x,0x7feb352d);x^=x>>>15;
    x=Math.imul(x,0x846ca68b);x^=x>>>16;return (x>>>0)/4294967296;
  }
  function domain(x,y,s,R){
    if(!Number.isFinite(x)||!Number.isFinite(y)||!Number.isFinite(s))
      throw new RangeError('Tracer integration produced a non-finite coordinate.');
    if(Math.hypot(x,y)>R || s<0 || s>1)
      throw new BoundaryCrossing('Tracer integration would leave the liquid domain.');
  }
  class Tracers {
    constructor(value,options={}){
      const o=observerOf(value,options.observation);validateObserver(o);
      const count=options.count??4096;
      if(!Number.isInteger(count)||count<1)throw new RangeError('Tracer count must be a positive integer.');
      this.count=count;this.R=o.R;
      this.initialSigma=finite(options.initialSigma??.98,'Initial tracer sigma');
      if(this.initialSigma<0||this.initialSigma>1)throw new RangeError('Initial tracer sigma must lie inside the liquid.');
      this.relativeTolerance=positive(options.relativeTolerance??1e-7,'Tracer relative tolerance');
      this.absolutePositionTolerance=positive(options.absolutePositionTolerance??o.R*1e-10,'Tracer absolute position tolerance');
      this.absoluteSigmaTolerance=positive(options.absoluteSigmaTolerance??1e-10,'Tracer absolute sigma tolerance');
      this.maxRefinements=options.maxRefinements??20;
      if(!Number.isInteger(this.maxRefinements)||this.maxRefinements<0)
        throw new RangeError('Tracer refinement limit must be a nonnegative integer.');
      this._coordinates=new Float64Array(3*count);this._next=new Float64Array(3*count);
      this._lastStep=null;
      this.reseed(o,{seed:options.seed??DEFAULT_SEED});
    }
    reseed(value,options={}){
      const o=observerOf(value);validateObserver(o);
      if(o.R!==this.R)throw new RangeError('Re-sprinkle cannot change the tracer domain radius.');
      const seed=seedValue(options.seed??((this.seed+1)>>>0)),angle=TAU*mixedSeed(seed);
      const next=new Float64Array(3*this.count);
      for(let p=0;p<this.count;p++){
        const r=this.R*Math.sqrt((p+.5)/this.count),theta=p*GOLDEN_ANGLE+angle;
        next[3*p]=r*Math.cos(theta);next[3*p+1]=r*Math.sin(theta);next[3*p+2]=this.initialSigma;
      }
      this._coordinates=next;this.seed=seed;this.time=o.time;this.generation=o.generation;
      this._lastStep=null;return this;
    }
    step(oldValue,newValue,dt){
      const old=observerOf(oldValue),next=observerOf(newValue);validateObserver(old,true);validateObserver(next,true);
      positive(dt,'Tracer physical time step');
      if(old.R!==this.R||next.R!==this.R)throw new RangeError('Tracer and fluid radii differ.');
      if(old.generation!==this.generation||next.generation!==this.generation)
        throw new RangeError('Tracer observations belong to a different solver generation.');
      if(old.time!==this.time)throw new RangeError('Tracer and old fluid physical times differ.');
      const elapsed=next.time-old.time;
      const roundoff=64*Number.EPSILON*Math.max(Math.abs(old.time),Math.abs(next.time),dt);
      if(!(elapsed>0)||Math.abs(elapsed-dt)>roundoff)
        throw new RangeError('Tracer dt does not match the fluid observation interval.');
      const k=new Float64Array(18),s0={},s1={},v0={x:0,y:0,z:0},v1={x:0,y:0,z:0};
      const stats={evaluations:0,acceptedSubsteps:0,refinements:0,maxAcceptedError:0,
        physicalDt:elapsed,method:'Cash-Karp RK5(4)',temporalField:'linear between every fluid step'};
      const R=this.R,rtol=this.relativeTolerance,atol=this.absolutePositionTolerance,
        stol=this.absoluteSigmaTolerance,maxRefinements=this.maxRefinements;
      function evaluate(x,y,s,alpha,offset){
        domain(x,y,s,R);
        let fx0=0,fy0=0,fs0=0,fx1=0,fy1=0,fs1=0;
        if(alpha!==1){
          old.surfaceAt(x,y,s0);positive(s0.H,'Old liquid height');
          old.velocityAt(x,y,s*s0.H,v0);
          fx0=v0.x;fy0=v0.y;fs0=(v0.z-s*(s0.Ht+v0.x*s0.Hx+v0.y*s0.Hy))/s0.H;
        }
        if(alpha!==0){
          next.surfaceAt(x,y,s1);positive(s1.H,'New liquid height');
          next.velocityAt(x,y,s*s1.H,v1);
          fx1=v1.x;fy1=v1.y;fs1=(v1.z-s*(s1.Ht+v1.x*s1.Hx+v1.y*s1.Hy))/s1.H;
        }
        k[offset]=(1-alpha)*fx0+alpha*fx1;
        k[offset+1]=(1-alpha)*fy0+alpha*fy1;
        k[offset+2]=(1-alpha)*fs0+alpha*fs1;
        if(!Number.isFinite(k[offset])||!Number.isFinite(k[offset+1])||!Number.isFinite(k[offset+2]))
          throw new RangeError('The fluid observation returned a non-finite tracer velocity.');
        stats.evaluations++;
      }
      function advance(x,y,s,begin,width,depth){
        let cx,cy,cs,error=Infinity,crossed=false;
        try {
          for(let stage=0;stage<6;stage++){
            let sx=x,sy=y,ss=s;
            for(let j=0;j<stage;j++){
              const q=width*B[stage][j];sx+=q*k[3*j];sy+=q*k[3*j+1];ss+=q*k[3*j+2];
            }
            evaluate(sx,sy,ss,(begin+A[stage]*width)/elapsed,3*stage);
          }
          cx=x;cy=y;cs=s;let ex=0,ey=0,es=0;
          for(let j=0;j<6;j++){
            const a=width*C5[j],e=width*(C5[j]-C4[j]);
            cx+=a*k[3*j];cy+=a*k[3*j+1];cs+=a*k[3*j+2];
            ex+=e*k[3*j];ey+=e*k[3*j+1];es+=e*k[3*j+2];
          }
          domain(cx,cy,cs,R);
          error=Math.max(Math.abs(ex)/(atol+rtol*Math.max(Math.abs(x),Math.abs(cx))),
            Math.abs(ey)/(atol+rtol*Math.max(Math.abs(y),Math.abs(cy))),
            Math.abs(es)/(stol+rtol*Math.max(Math.abs(s),Math.abs(cs))));
        } catch(e){if(e instanceof BoundaryCrossing)crossed=true;else throw e;}
        if(!crossed&&error<=1){
          stats.acceptedSubsteps++;stats.maxAcceptedError=Math.max(stats.maxAcceptedError,error);
          return [cx,cy,cs];
        }
        if(depth>=maxRefinements||begin+width/2===begin||begin+width/2===begin+width)
          throw new RangeError(crossed
            ?'Tracer integration would leave the liquid domain after refinement; no boundary clipping was applied.'
            :'Tracer integration could not meet its requested error tolerance within the refinement limit.');
        stats.refinements++;
        const half=width/2,left=advance(x,y,s,begin,half,depth+1);
        return advance(left[0],left[1],left[2],begin+half,half,depth+1);
      }
      // Writes go to a spare array. Failure anywhere leaves the entire prior
      // snapshot, seed, and clock available to the caller for an honest error.
      for(let p=0;p<this.count;p++){
        const i=3*p,a=advance(this._coordinates[i],this._coordinates[i+1],this._coordinates[i+2],0,elapsed,0);
        this._next[i]=a[0];this._next[i+1]=a[1];this._next[i+2]=a[2];
      }
      const previous=this._coordinates;this._coordinates=this._next;this._next=previous;
      this.time=next.time;this._lastStep=Object.freeze(stats);return this;
    }
    physicalPositions(value){
      const o=observerOf(value);validateObserver(o);
      if(o.R!==this.R||o.time!==this.time||o.generation!==this.generation)
        throw new RangeError('Particle positions require the matching fluid frame and physical time.');
      const out=new Float64Array(this._coordinates.length),surface={};
      for(let p=0;p<this.count;p++){
        const i=3*p,x=this._coordinates[i],y=this._coordinates[i+1],s=this._coordinates[i+2];
        domain(x,y,s,this.R);
        const H=typeof o.heightAt==='function'?o.heightAt(Math.hypot(x,y),Math.atan2(y,x)):
          (o.surfaceAt(x,y,surface),surface.H);
        positive(H,'Liquid height');out[i]=x;out[i+1]=y;out[i+2]=s*H;
      }
      return out;
    }
    snapshot(){
      return Object.freeze({format:'cell3d-tracers-v1',count:this.count,seed:this.seed,time:this.time,
        generation:this.generation,R:this.R,initialSigma:this.initialSigma,
        coordinates:Float64Array.from(this._coordinates),coordinateSystem:'x[m],y[m],sigma=z/H',
        interpretation:'passive flow tracers; not finite sand-grain dynamics',
        relativeTolerance:this.relativeTolerance,absolutePositionTolerance:this.absolutePositionTolerance,
        absoluteSigmaTolerance:this.absoluteSigmaTolerance,maxRefinements:this.maxRefinements,
        lastStep:this._lastStep?{...this._lastStep}:null});
    }
  }
  return Object.freeze({Tracers,create:(value,options)=>new Tracers(value,options)});
})();
if(typeof module!=='undefined'&&module.exports)module.exports=FARADAY_CELL3D_TRACERS;
if(typeof globalThis!=='undefined')globalThis.FARADAY_CELL3D_TRACERS=FARADAY_CELL3D_TRACERS;
