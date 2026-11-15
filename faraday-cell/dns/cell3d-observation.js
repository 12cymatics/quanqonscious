'use strict';

/* Observations of the nonlinear solver, in metres and seconds. This module does
 * not step the fluid or feed anything back into it. Every observer owns copies
 * of one complete field: moving a camera, changing display scale, or retaining
 * a frame therefore cannot change a solver array or combine different times.
 *
 * Velocity reconstruction respects each component's staggered family. Columns
 * are reconstructed at a common physical z, not at a common sigma, before the
 * radial and periodic angular interpolations. The vertical polynomial includes
 * the actual zero-velocity floor boundary. The sidewall is likewise a boundary
 * value. Inside the first radial ring, a regular Cartesian patch matches that
 * ring while supplying one finite axis value and gradient. Native sample
 * heights still use the solver's own metric and antipodal continuation.
 * This is an observation interpolant, not a new momentum discretization.
 */
const FARADAY_CELL3D_OBSERVATION = (() => {
  const TAU = 2*Math.PI;
  const ARRAY_KEYS = ['rf', 'rc', 'sf', 'sc', 'eta', 'Ht', 'u', 'v', 'w'];
  const wrap = (k, n) => ((k % n) + n) % n;
  function finite(value, name){
    if (typeof value !== 'number' || !Number.isFinite(value))
      throw new TypeError(`${name} must be finite.`);
    return value;
  }
  function integer(value, name, minimum){
    if (!Number.isInteger(value) || value < minimum)
      throw new RangeError(`${name} must be an integer at least ${minimum}.`);
    return value;
  }
  function copyArray(value, length, name){
    if (!(value instanceof Float64Array) || value.length !== length)
      throw new TypeError(`${name} must be a Float64Array of length ${length}.`);
    const out = Float64Array.from(value);
    for (let i=0; i<out.length; i++) if (!Number.isFinite(out[i]))
      throw new RangeError(`${name}[${i}] is not finite.`);
    return out;
  }
  function monotone(a, name){
    for(let i=1; i<a.length; i++) if (!(a[i] > a[i-1]))
      throw new RangeError(`${name} must be strictly increasing.`);
  }
  function capture(S, options={}){
    if (!S) throw new TypeError('A complete nonlinear solver state is required.');
    const source = options.source || S;
    const nr=integer(S.nr,'nr',4), nth=integer(S.nth,'nth',8), nz=integer(S.nz,'nz',4);
    if(nth%2) throw new RangeError('The azimuthal grid must have an even size.');
    const R=finite(S.R,'R'), h=finite(S.h,'h');
    if(!(R>0 && h>0)) throw new RangeError('Radius and depth must be positive.');
    const time=finite(options.time ?? S.t ?? S.time ?? 0,'time');
    if(time<0) throw new RangeError('Physical time cannot be negative.');
    const generation=integer(options.generation ?? S.generation ?? 0,'generation',0);
    const steps=integer(options.steps ?? S.steps ?? 0,'steps',0);
    const contact=S.contact;
    if(contact!=='pinned' && contact!=='free') throw new TypeError('Unknown contact boundary.');
    const data={format:'cell3d-observation-v1',nr,nth,nz,R,h,time,generation,steps,
      initial:options.initial === undefined ? !!S.initial : !!options.initial,contact,
      rf:copyArray(S.rf,nr+1,'rf'),rc:copyArray(S.rc,nr,'rc'),
      sf:copyArray(S.sf,nz+1,'sf'),sc:copyArray(S.sc,nz,'sc'),
      eta:copyArray(source.eta,nr*nth,'eta'),Ht:copyArray(source.Ht,nr*nth,'Ht'),
      u:copyArray(source.u,(nr+1)*nth*nz,'u'),v:copyArray(source.v,nr*nth*nz,'v'),
      w:copyArray(source.w,nr*nth*(nz+1),'w')};
    for(const name of ['rf','rc','sf','sc']) monotone(data[name],name);
    if(data.rf[0]!==0 || data.rf[nr]!==R || data.sf[0]!==0 || data.sf[nz]!==1)
      throw new RangeError('Grid endpoints do not match the physical domain.');
    for(let i=0;i<nr;i++) if(!(data.rf[i]<data.rc[i] && data.rc[i]<data.rf[i+1]))
      throw new RangeError('A radial centre lies outside its cell.');
    for(let j=0;j<nz;j++) if(!(data.sf[j]<data.sc[j] && data.sc[j]<data.sf[j+1]))
      throw new RangeError('A vertical centre lies outside its cell.');
    for(let e=0;e<data.eta.length;e++) if(!(h+data.eta[e]>0))
      throw new RangeError('The surface reaches the floor; an observation cannot repair it.');
    return Object.freeze(data);
  }

  function cloneSnapshot(data){
    const out={...data};
    for(const key of ARRAY_KEYS) out[key]=Float64Array.from(data[key]);
    return Object.freeze(out);
  }

  // Locate a bracket. Callers explicitly supply physical boundary values when
  // a component has no node on that boundary; the query itself is never clipped.
  function bracket(nodes, value){
    let lo=0,hi=nodes.length-1;
    while(hi-lo>1){const m=(lo+hi)>>1;if(nodes[m]<=value)lo=m;else hi=m;}
    return lo;
  }
  function createObserver(S, options={}){
    // capture also validates a previously packed snapshot and copies its arrays.
    const d=capture(S,options),{nr,nth,nz,R,h,rf,rc,sf,sc,eta,Ht}=d,dth=TAU/nth;
    const ie=(i,k)=>i*nth+wrap(k,nth);
    const extendedR=new Float64Array(nr+2);
    extendedR[0]=-rc[0];extendedR.set(rc,1);extendedR[nr+1]=R;
    function axisParameters(field){
      let mean0=0,mean1=0,gx=0,gy=0;
      for(let k=0;k<nth;k++){
        const th=(k+.5)*dth,value=field[k];
        mean0+=value/nth;mean1+=field[nth+k]/nth;
        gx+=2*value*Math.cos(th)/(nth*rc[0]);gy+=2*value*Math.sin(th)/(nth*rc[0]);
      }
      const r0=rc[0],r1=rc[1];
      return {value:(r1*r1*mean0-r0*r0*mean1)/(r1*r1-r0*r0),gx,gy};
    }
    const axisFields=new Map([[eta,axisParameters(eta)],[Ht,axisParameters(Ht)]]);
    if(!(h+axisFields.get(eta).value>0))throw new RangeError('Reconstructed axis surface reaches the floor.');
    function axisSurface(field,r,theta){
      const axis=axisFields.get(field),c=Math.cos(theta),s=Math.sin(theta),r0=rc[0];
      const tt=theta/dth-.5,k=Math.floor(tt),g=tt-k;
      const a=field[ie(0,k)],b=field[ie(0,k+1)],edge=(1-g)*a+g*b;
      const linear=axis.gx*c+axis.gy*s,tangent=-axis.gx*s+axis.gy*c;
      const remainder=edge-axis.value-r0*linear,q=r/r0;
      // Match the entire first-ring interpolant, but make every unresolved
      // angular remainder vanish quadratically toward the unique axis value.
      // Its first angular harmonic supplies a Cartesian linear term. Thus both
      // value and gradient have finite unique axis limits, including even m.
      return {value:axis.value+r*linear+q*q*remainder,
        dr:linear+2*r*remainder/(r0*r0),
        dtheta:r*tangent+q*q*((b-a)/dth-r0*tangent)};
    }
    function surfaceScalar(field,r,theta,regularAxis=true){
      finite(r,'r');finite(theta,'theta');
      if(r<0 || r>R) throw new RangeError('Surface query is outside the cylinder.');
      if(regularAxis && r<rc[0])return axisSurface(field,r,theta).value;
      const a=bracket(extendedR,r),r0=extendedR[a],r1=extendedR[a+1];
      const f=(r-r0)/(r1-r0),tt=theta/dth-.5,k=Math.floor(tt),g=tt-k;
      const value=(ring,column)=>ring===0 ? field[ie(0,column+nth/2)]
        :ring===nr+1 ? d.contact==='pinned'?0:field[ie(nr-1,column)]
        :field[ie(ring-1,column)];
      return (1-f)*((1-g)*value(a,k)+g*value(a,k+1))
        +f*((1-g)*value(a+1,k)+g*value(a+1,k+1));
    }
    const etaAt=(r,theta)=>surfaceScalar(eta,r,theta);
    const heightAt=(r,theta)=>h+etaAt(r,theta);
    function surfaceAt(x,y,out={H:0,Ht:0,Hx:0,Hy:0}){
      finite(x,'x');finite(y,'y');
      const r=Math.hypot(x,y),theta=Math.atan2(y,x);
      if(r>R)throw new RangeError('Surface query is outside the cylinder.');
      if(r===0){
        const a=axisFields.get(eta);
        out.H=h+a.value;out.Ht=axisFields.get(Ht).value;
        out.Hx=a.gx;out.Hy=a.gy;return out;
      }
      if(r<rc[0]){
        const a=axisSurface(eta,r,theta),c=x/r,s=y/r;
        out.H=h+a.value;out.Ht=axisSurface(Ht,r,theta).value;
        out.Hx=a.dr*c-a.dtheta*s/r;out.Hy=a.dr*s+a.dtheta*c/r;return out;
      }
      const a=bracket(extendedR,r),r0=extendedR[a],r1=extendedR[a+1];
      const f=(r-r0)/(r1-r0),tt=theta/dth-.5,k=Math.floor(tt),g=tt-k;
      const value=(ring,column)=>ring===0 ? eta[ie(0,column+nth/2)]
        :ring===nr+1 ? d.contact==='pinned'?0:eta[ie(nr-1,column)]
        :eta[ie(ring-1,column)];
      const e00=value(a,k),e01=value(a,k+1),e10=value(a+1,k),e11=value(a+1,k+1);
      const dr=((1-g)*(e10-e00)+g*(e11-e01))/(r1-r0);
      const dt=((1-f)*(e01-e00)+f*(e11-e10))/dth,c=x/r,s=y/r;
      out.H=h+(1-f)*((1-g)*e00+g*e01)+f*((1-g)*e10+g*e11);
      out.Ht=surfaceScalar(Ht,r,theta);out.Hx=dr*c-dt*s/r;out.Hy=dr*s+dt*c/r;
      return out;
    }
    const families={
      u:{field:d.u,rn:rf,sn:sc,thOff:.5,axisSign:-1,stride:nz,hasFloor:false},
      v:{field:d.v,rn:rc,sn:sc,thOff:0,axisSign:-1,stride:nz,hasFloor:false},
      w:{field:d.w,rn:rc,sn:sf,thOff:.5,axisSign:1,stride:nz+1,hasFloor:true}
    };
    for(const fam of Object.values(families)){
      const rows=fam.rn.length;fam.H=new Float64Array(rows*nth);
      for(let a=0;a<rows;a++)for(let k=0;k<nth;k++)
        // Stored sigma nodes belong to the solver's native column geometry,
        // including its axis continuation. Regularization is only for queried
        // observations, never a relocation of the captured velocity samples.
        fam.H[a*nth+k]=h+surfaceScalar(eta,fam.rn[a],(k+fam.thOff)*dth,false);
      // u and v have no floor node. Include their prescribed no-slip value in
      // the reconstruction rather than extrapolating a nonzero floor velocity.
      fam.znodes=fam.hasFloor?fam.sn:Float64Array.of(0,...fam.sn);
      fam.polynomials=new Map();fam.stencilCount=fam.znodes.length-3;
      fam.stencils=[];
      for(let start=0;start<fam.stencilCount;start++){
        const a=fam.znodes[start],b=fam.znodes[start+1],c=fam.znodes[start+2],e=fam.znodes[start+3];
        fam.stencils.push({a,b,c,ab:1/(b-a),bc:1/(c-b),ce:1/(e-c),
          ac:1/(c-a),be:1/(e-b),ae:1/(e-a)});
      }
    }
    function column(fam,a,k,z){
      let sign=1;
      if(a<0){a=-1-a;k+=nth/2;sign=fam.axisSign;}
      k=wrap(k,nth);
      if(a>=fam.rn.length || fam.rn[a]===R || z===0)return 0;
      const sigma=z/fam.H[a*nth+k],nodes=fam.znodes;
      const j=bracket(nodes,sigma),count=Math.min(4,nodes.length);
      let start=j-1;if(start<0)start=0;if(start+count>nodes.length)start=nodes.length-count;
      const cacheKey=(a*nth+k)*fam.stencilCount+start;
      let polynomial=fam.polynomials.get(cacheKey);
      const s=fam.stencils[start];
      if(!polynomial){
        const value=n=>n===0?0:fam.field[(a*nth+k)*fam.stride+(fam.hasFloor?n:n-1)];
        const y0=value(start),y1=value(start+1),y2=value(start+2),y3=value(start+3);
        const d01=(y1-y0)*s.ab,d12=(y2-y1)*s.bc,d23=(y3-y2)*s.ce;
        const d012=(d12-d01)*s.ac,d123=(d23-d12)*s.be;
        polynomial=Float64Array.of(y0,d01,d012,(d123-d012)*s.ae);
        fam.polynomials.set(cacheKey,polynomial);
      }
      // The same four-node interpolation polynomial, evaluated in Newton form.
      // Cached coefficients belong to this immutable snapshot, not a live field.
      return sign*(polynomial[0]+(sigma-s.a)*(polynomial[1]
        +(sigma-s.b)*(polynomial[2]+(sigma-s.c)*polynomial[3])));
    }
    function component(fam,r,theta,z){
      const rn=fam.rn;
      let a,b,r0,r1;
      if(r<rn[0]){a=-1;b=0;r0=-rn[0];r1=rn[0];}
      else if(r>=rn[rn.length-1]){
        if(rn[rn.length-1]===R){a=rn.length-2;b=a+1;r0=rn[a];r1=R;}
        else {a=rn.length-1;b=rn.length;r0=rn[a];r1=R;}
      }else {a=bracket(rn,r);b=a+1;r0=rn[a];r1=rn[b];}
      const f=(r-r0)/(r1-r0),tt=theta/dth-fam.thOff,k=Math.floor(tt),g=tt-k;
      return (1-f)*((1-g)*column(fam,a,k,z)+g*column(fam,a,k+1,z))
        +f*((1-g)*column(fam,b,k,z)+g*column(fam,b,k+1,z));
    }
    function ordinaryVelocity(r,theta,z,out){
      const ur=component(families.u,r,theta,z),ut=component(families.v,r,theta,z);
      const c=Math.cos(theta),s=Math.sin(theta);
      out.x=ur*c-ut*s;out.y=ur*s+ut*c;out.z=component(families.w,r,theta,z);return out;
    }
    function innerVelocity(r,theta,z,out){
      let ux=0,uy=0,w0=0,w1=0;
      const r0=rc[0],r1=rc[1];
      for(let k=0;k<nth;k++){
        const th=(k+.5)*dth,u=column(families.u,0,k,z);
        ux+=2*u*Math.cos(th)/nth;uy+=2*u*Math.sin(th)/nth;
        w0+=component(families.w,r0,th,z)/nth;w1+=component(families.w,r1,th,z)/nth;
      }
      const center={x:ux,y:uy,z:(r1*r1*w0-r0*r0*w1)/(r1*r1-r0*r0)};
      if(r===0){out.x=center.x;out.y=center.y;out.z=center.z;return out;}
      const gx={x:0,y:0,z:0},gy={x:0,y:0,z:0},v={x:0,y:0,z:0};
      for(let k=0;k<nth;k++){
        const th=(k+.5)*dth;ordinaryVelocity(r0,th,z,v);
        const c=2*Math.cos(th)/(nth*r0),s=2*Math.sin(th)/(nth*r0);
        for(const key of ['x','y','z']){gx[key]+=(v[key]-center[key])*c;gy[key]+=(v[key]-center[key])*s;}
      }
      ordinaryVelocity(r0,theta,z,v);
      const c=Math.cos(theta),s=Math.sin(theta),q=r/r0;
      for(const key of ['x','y','z']){
        const linear=gx[key]*c+gy[key]*s;
        out[key]=center[key]+r*linear+q*q*(v[key]-center[key]-r0*linear);
      }
      return out;
    }
    function velocityAt(x,y,z,out={x:0,y:0,z:0}){
      finite(x,'x');finite(y,'y');finite(z,'z');
      const r=Math.hypot(x,y),theta=Math.atan2(y,x);
      if(r>R || z<0 || z>heightAt(r,theta))
        throw new RangeError('Velocity query is outside the liquid domain.');
      if(r===R || z===0){out.x=out.y=out.z=0;return out;}
      if(r<rc[0])return innerVelocity(r,theta,z,out);
      return ordinaryVelocity(r,theta,z,out);
    }
    function volumeDisplacement(x,y,z){
      const r=Math.hypot(x,y),theta=Math.atan2(y,x),e=etaAt(r,theta),H=h+e;
      if(z<0 || z>H)throw new RangeError('Particle is outside the liquid domain.');
      return z*e/H;
    }
    function frame(frameOptions={}){
      const glyphLimit=integer(frameOptions.maxFlowGlyphs ?? 384,'maxFlowGlyphs',0);
      const vertexCount=1+(nr+1)*nth,positions=new Float32Array(vertexCount*3);
      const surfaceDisplacement=new Float32Array(vertexCount);
      const axisEta=etaAt(0,0);let minEta=Infinity,maxEta=-Infinity;
      positions[2]=h+axisEta;surfaceDisplacement[0]=axisEta;
      for(let ring=0;ring<=nr;ring++)for(let k=0;k<nth;k++){
        const r=ring===nr?R:rc[ring],th=(k+.5)*dth,e=etaAt(r,th),v=1+ring*nth+k;
        positions[3*v]=r*Math.cos(th);positions[3*v+1]=r*Math.sin(th);positions[3*v+2]=h+e;
        surfaceDisplacement[v]=e;minEta=Math.min(minEta,e);maxEta=Math.max(maxEta,e);
      }
      minEta=Math.min(minEta,axisEta);maxEta=Math.max(maxEta,axisEta);
      const indices=new Uint32Array(nth*(1+2*nr)*3);let q=0;
      for(let k=0;k<nth;k++){indices[q++]=0;indices[q++]=1+k;indices[q++]=1+(k+1)%nth;}
      for(let ring=0;ring<nr;ring++)for(let k=0;k<nth;k++){
        const a=1+ring*nth+k,b=1+ring*nth+(k+1)%nth,c=a+nth,e=b+nth;
        indices[q++]=a;indices[q++]=c;indices[q++]=e;
        indices[q++]=a;indices[q++]=e;indices[q++]=b;
      }
      let sr=1,st=1,sz=1;
      const count=()=>Math.ceil(nr/sr)*Math.ceil(nth/st)*Math.ceil(nz/sz);
      while(glyphLimit && count()>glyphLimit){
        const ar=nr/sr,at=nth/st,az=nz/sz;
        if(ar>=at && ar>=az)sr++;else if(at>=az)st++;else sz++;
      }
      const glyphCount=glyphLimit?count():0,flowPositions=new Float32Array(glyphCount*3);
      const flowVectors=new Float32Array(glyphCount*3),flowDisplacement=new Float32Array(glyphCount);
      let g=0,maxSpeed=0;const velocity={x:0,y:0,z:0};
      if(glyphLimit)for(let i=0;i<nr;i+=sr)for(let k=0;k<nth;k+=st)for(let j=0;j<nz;j+=sz){
        const r=rc[i],th=(k+.5)*dth,e=etaAt(r,th),z=sc[j]*(h+e),x=r*Math.cos(th),y=r*Math.sin(th);
        velocityAt(x,y,z,velocity);
        flowPositions[3*g]=x;flowPositions[3*g+1]=y;flowPositions[3*g+2]=z;
        flowVectors[3*g]=velocity.x;flowVectors[3*g+1]=velocity.y;flowVectors[3*g+2]=velocity.z;
        flowDisplacement[g]=sc[j]*e;
        maxSpeed=Math.max(maxSpeed,Math.hypot(velocity.x,velocity.y,velocity.z));g++;
      }
      let particles,particleDisplacement;
      if(frameOptions.particles!==undefined){
        const input=frameOptions.particles;
        if(!(input instanceof Float64Array) || input.length%3)
          throw new TypeError('Particle positions must be Float64Array xyz triples in metres.');
        particles=copyArray(input,input.length,'particles');particleDisplacement=new Float32Array(input.length/3);
        for(let p=0;p<particleDisplacement.length;p++)
          particleDisplacement[p]=volumeDisplacement(particles[3*p],particles[3*p+1],particles[3*p+2]);
      }
      return Object.freeze({R,h,time:d.time,generation:d.generation,steps:d.steps,initial:d.initial,
        positions,indices,surfaceDisplacement,particles,particleDisplacement,
        flowPositions,flowVectors,flowDisplacement,etaRange:[minEta,maxEta],maxSpeed,
        maxSpeedScope:'emitted-flow-glyphs',grid:{nr,nth,nz},
        flowSampling:{radialStride:sr,azimuthalStride:st,verticalStride:sz,
          count:glyphCount,totalCells:nr*nth*nz,displayOnly:true},
        units:{position:'m',velocity:'m/s',time:'s'},
        provenance:{model:'nonlinear-incompressible-free-surface-navier-stokes',
          fieldKind:d.initial?'initial-condition':'evolved-solver-frame',
          generation:d.generation,physicalTime:d.time,steps:d.steps}});
    }
    return Object.freeze({R,h,time:d.time,generation:d.generation,steps:d.steps,initial:d.initial,
      etaAt,heightAt,surfaceAt,etaDotAt:(r,theta)=>surfaceScalar(Ht,r,theta),velocityAt,
      volumeDisplacement,frame,snapshot:()=>cloneSnapshot(d)});
  }
  const createFrame=(S,options={})=>createObserver(S,options).frame(options);
  function createInitialFrame(options){
    const {at}=options,R=finite(options.R,'R'),h=finite(options.h,'h');
    if(!(R>0 && h>0) || typeof at!=='function')
      throw new TypeError('Positive geometry and an analytic initial surface are required.');
    const generation=integer(options.generation ?? 0,'generation',0);
    // This is solely the tessellation of the explicitly specified initial
    // function. The nonlinear solver chooses its own accuracy grid later.
    const nr=integer(options.radialSegments ?? 48,'radialSegments',2);
    const nth=integer(options.azimuthalSegments ?? 96,'azimuthalSegments',3);
    const elevation=(r,theta)=>finite(at(r/R,theta).Re,'initial elevation');
    const positions=new Float32Array((1+(nr+1)*nth)*3);
    const surfaceDisplacement=new Float32Array(1+(nr+1)*nth);
    const axis=elevation(0,0);positions[2]=h+axis;surfaceDisplacement[0]=axis;
    if(!(h+axis>0))throw new RangeError('Initial surface reaches the floor.');
    let lo=axis,hi=axis;
    for(let ring=0;ring<=nr;ring++)for(let k=0;k<nth;k++){
      const r=ring===nr?R:R*(ring+.5)/nr,th=(k+.5)*TAU/nth;
      const e=elevation(r,th),v=1+ring*nth+k;
      if(!(h+e>0))throw new RangeError('Initial surface reaches the floor.');
      positions[3*v]=r*Math.cos(th);positions[3*v+1]=r*Math.sin(th);positions[3*v+2]=h+e;
      surfaceDisplacement[v]=e;lo=Math.min(lo,e);hi=Math.max(hi,e);
    }
    const indices=new Uint32Array(nth*(1+2*nr)*3);let q=0;
    for(let k=0;k<nth;k++){indices[q++]=0;indices[q++]=1+k;indices[q++]=1+(k+1)%nth;}
    for(let ring=0;ring<nr;ring++)for(let k=0;k<nth;k++){
      const a=1+ring*nth+k,b=1+ring*nth+(k+1)%nth,c=a+nth,e=b+nth;
      indices[q++]=a;indices[q++]=c;indices[q++]=e;indices[q++]=a;indices[q++]=e;indices[q++]=b;
    }
    let particles,particleDisplacement;
    if(options.particles!==undefined){
      if(!(options.particles instanceof Float64Array) || options.particles.length%3)
        throw new TypeError('Particle positions must be Float64Array xyz triples in metres.');
      particles=copyArray(options.particles,options.particles.length,'particles');
      particleDisplacement=new Float32Array(particles.length/3);
      for(let p=0;p<particles.length;p+=3){
        const r=Math.hypot(particles[p],particles[p+1]),theta=Math.atan2(particles[p+1],particles[p]);
        if(r>R)throw new RangeError('Initial particle lies outside the cylinder.');
        const e=elevation(r,theta),H=h+e,z=particles[p+2];
        if(z<0 || z>H)throw new RangeError('Initial particle lies outside the liquid.');
        particleDisplacement[p/3]=z*e/H;
      }
    }
    return Object.freeze({R,h,time:0,generation,steps:0,initial:true,
      positions,indices,surfaceDisplacement,particles,particleDisplacement,
      flowPositions:new Float32Array(0),flowVectors:new Float32Array(0),flowDisplacement:new Float32Array(0),
      etaRange:[lo,hi],maxSpeed:0,maxSpeedScope:'specified-zero-initial-velocity',grid:null,
      displayTessellation:{radialSegments:nr,azimuthalSegments:nth,selectsSolverGrid:false},
      flowSampling:{count:0,totalCells:0,displayOnly:true},units:{position:'m',velocity:'m/s',time:'s'},
      provenance:{model:'nonlinear-incompressible-free-surface-navier-stokes',fieldKind:'initial-condition',
        generation,physicalTime:0,steps:0,initialCondition:options.initialCondition
          ? JSON.parse(JSON.stringify(options.initialCondition)):undefined}});
  }
  return Object.freeze({capture,createObserver,createFrame,createInitialFrame});
})();
if(typeof module!=='undefined' && module.exports)module.exports=FARADAY_CELL3D_OBSERVATION;
if(typeof globalThis!=='undefined')globalThis.FARADAY_CELL3D_OBSERVATION=FARADAY_CELL3D_OBSERVATION;
