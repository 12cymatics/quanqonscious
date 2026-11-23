"use strict";

/* Independent exact arithmetic controls for the comparison experiment only.
 * The returned source is inserted into the existing polynomial closure before
 * `const api=`. It changes named bindings only in that isolated VM, never the
 * production file. Inputs and outputs retain the repository's exact API.
 * Conventional helpers use only Q, poly, and elementary rational arithmetic;
 * they do not call the original Vedic polynomial algorithms.
 */

const names = Object.freeze([
  "ekadhikena", "nikhilam", "urdhva", "paravartya", "shunyamCoefficient",
  "nodalWeights", "symmetricTranslate", "purana", "calana", "yavadunam",
  "vyashti", "sesanyankena", "sopantyaProductDerivative", "ekanyunena",
  "gunitasamuccaya", "gunakasamuccaya", "anurupyena", "shishyateKnownLinear",
  "adyamDivide", "kevalaih", "osculateEvaluate", "tavadunamCube",
  "extendedSquare", "antyayordasakepi", "antyayorevaProductSecond",
  "coefficientSum", "retainedQuadratic", "reciprocalPolynomial",
  "gunitasamuccayaSamuccayagunitah"
]);

const helpers = String.raw`
function __conventionalMul(a,b){
  a=poly(a);b=poly(b);
  const out=Array(a.length+b.length-1).fill(ZERO);
  for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)
    out[i+j]=add(out[i+j],mul(a[i],b[j]));
  return poly(out);
}
function __conventionalAdd(a,b){
  a=poly(a);b=poly(b);
  return poly(Array.from({length:Math.max(a.length,b.length)},(_,i)=>add(a[i]||ZERO,b[i]||ZERO)));
}
function __conventionalScale(p,s){p=poly(p);s=Q(s);return poly(p.map(v=>mul(v,s)));}
function __conventionalProduct(factors){
  if(!Array.isArray(factors)||!factors.length)throw new RangeError("At least one factor is required");
  let out=poly([ONE]);
  for(const factor of factors)out=__conventionalMul(out,poly(factor));
  return out;
}
function __conventionalPower(p,k){
  p=poly(p);
  if(!Number.isSafeInteger(k)||k<0)throw new RangeError("Polynomial exponent must be a nonnegative safe integer");
  let out=poly([ONE]);
  for(let i=0;i<k;i++)out=__conventionalMul(out,p);
  return out;
}
function __conventionalChoose(n,k){
  if(k<0||k>n)return 0n;
  let out=1n;
  for(let j=1;j<=k;j++)out=out*BigInt(n-j+1)/BigInt(j);
  return out;
}
function __conventionalDerivative(p,order=1){
  p=poly(p);
  if(!Number.isSafeInteger(order)||order<0)throw new RangeError("Derivative order must be nonnegative");
  if(order>=p.length)return poly([ZERO]);
  const out=[];
  for(let k=order;k<p.length;k++){
    let falling=1n;
    for(let j=0;j<order;j++)falling*=BigInt(k-j);
    out.push(mul(p[k],falling));
  }
  return poly(out);
}
function __conventionalTranslate(p,shift){
  p=poly(p);shift=Q(shift);
  const out=Array(p.length).fill(ZERO);
  for(let n=0;n<p.length;n++)for(let k=0;k<=n;k++)
    out[k]=add(out[k],mul(p[n],mul(__conventionalChoose(n,k),pow(shift,BigInt(n-k)))));
  return poly(out);
}
function __conventionalEvaluate(p,x){
  p=poly(p);x=Q(x);let out=ZERO;
  for(let k=p.length-1;k>=0;k--)out=add(mul(out,x),p[k]);
  return out;
}
function __conventionalDivide(a,b){
  a=poly(a);b=poly(b);
  if(b.length===1&&b[0].n===0n)throw new RangeError("Zero divisor polynomial");
  if(a.length<b.length)return Object.freeze({quotient:poly([ZERO]),remainder:a});
  const q=Array(a.length-b.length+1).fill(ZERO);
  let remainder=a;
  while(!(remainder.length===1&&remainder[0].n===0n)&&remainder.length>=b.length){
    const shift=remainder.length-b.length,coefficient=div(remainder[remainder.length-1],b[b.length-1]);
    q[shift]=add(q[shift],coefficient);
    const next=Array.from(remainder);
    for(let j=0;j<b.length;j++)next[j+shift]=sub(next[j+shift],mul(coefficient,b[j]));
    remainder=poly(next);
  }
  return Object.freeze({quotient:poly(q),remainder});
}
function __conventionalKnownLinear(a,b){
  a=poly(a);b=poly(b);
  if(a.length<2||b.length!==2||b[0].n===0n)
    throw new RangeError("A positive-degree dividend and known affine factor with nonzero endpoints are required");
  const result=__conventionalDivide(a,b);
  if(result.remainder.some(c=>c.n!==0n))throw new RangeError("The proposed binomial is not a factor");
  return result.quotient;
}
function __conventionalInteger(x){x=Q(x);if(x.d!==1n)throw new TypeError("An exact integer is required");return x.n;}
function __conventionalRecurring(numerator,denominator){
  let remainder=numerator;
  const initial=remainder,digits=[];
  do{
    const expanded=10n*remainder;
    digits.push(Number(expanded/denominator));
    remainder=expanded%denominator;
  }while(remainder!==initial);
  return Object.freeze(digits);
}
function __conventionalNodes(xs){
  if(!Array.isArray(xs)||!xs.length)throw new RangeError("At least one interpolation node is required");
  xs=xs.map(x=>Q(x));
  for(let i=0;i<xs.length;i++)for(let j=0;j<i;j++)
    if(eq(xs[i],xs[j]))throw new RangeError("Interpolation nodes must be distinct");
  return xs;
}
`;

const bodies = Object.freeze({
  1: String.raw`function(denominator){
    const d=__conventionalInteger(denominator);
    if(d<9n||d%10n!==9n)throw new RangeError("Ekadhikena requires a positive denominator ending in nine");
    return __conventionalRecurring(1n,d);
  }`,
  2: String.raw`function(a,b,base){
    a=poly(a);b=poly(b);poly(base);
    return __conventionalMul(a,b);
  }`,
  3: String.raw`function(a,b){return __conventionalMul(a,b);}`,
  4: String.raw`function(a,b){return __conventionalDivide(a,b);}`,
  5: String.raw`function(values,index,total=ZERO){
    if(!Array.isArray(values)||!values.length||!Number.isSafeInteger(index)||index<0||index>=values.length)
      throw new RangeError("A valid coefficient index is required");
    const out=values.map((x,i)=>i===index?ZERO:Q(x));
    let missing=Q(total);
    for(let i=0;i<out.length;i++)if(i!==index)missing=sub(missing,out[i]);
    out[index]=missing;
    return Object.freeze(out);
  }`,
  6: String.raw`function(xs,x){
    xs=__conventionalNodes(xs);x=Q(x);
    return Object.freeze(xs.map((node,i)=>{
      let value=ONE;
      for(let j=0;j<xs.length;j++)if(i!==j)value=mul(value,div(sub(x,xs[j]),sub(node,xs[j])));
      return value;
    }));
  }`,
  7: String.raw`function(p,center){
    const shifted=__conventionalTranslate(p,center);
    const even=poly(shifted.map((c,i)=>i%2?ZERO:c));
    const odd=poly(shifted.map((c,i)=>i%2?c:ZERO));
    return Object.freeze({even,odd,sum:__conventionalScale(even,TWO),difference:__conventionalScale(odd,TWO)});
  }`,
  8: String.raw`function(a,b,c){
    a=Q(a);b=Q(b);c=Q(c);
    if(a.n===0n)throw new RangeError("Quadratic leading coefficient must be nonzero");
    const center=div(neg(b),mul(TWO,a));
    const centered=__conventionalTranslate([c,b,a],center);
    const offset=centered[0],radicand=div(neg(offset),a);
    return Object.freeze({leading:a,center,offset,radicand,centered});
  }`,
  9: String.raw`function(p,order=1){return __conventionalDerivative(p,order);}`,
  10: String.raw`function(p,base){p=poly(p);poly(base);return __conventionalPower(p,2);}`,
  11: String.raw`function(a,b,total=0n){
    a=Q(a);b=Q(b);total=Q(total);
    const middle=div(add(a,b),TWO),halfDifference=div(sub(a,b),TWO);
    let polynomial=__conventionalAdd(__conventionalPower([a,ONE],4),__conventionalPower([b,ONE],4));
    polynomial=__conventionalAdd(polynomial,[neg(total)]);
    const centered=__conventionalTranslate(polynomial,neg(middle));
    const squarePolynomial=poly([centered[0]||ZERO,centered[2]||ZERO,centered[4]||ZERO]);
    return Object.freeze({middle,halfDifference,squarePolynomial,polynomial});
  }`,
  12: String.raw`function(denominator){
    const d=__conventionalInteger(denominator);
    if(d<=1n||gcd(d,10n)!==1n)throw new RangeError("Remainder digits require denominator greater than one and coprime to ten");
    return __conventionalRecurring(1n,d);
  }`,
  13: String.raw`function(first,step){
    const a=poly(first),d=Q(step);
    if(a.length!==2||!eq(a[1],ONE)||d.n===0n)
      throw new RangeError("A monic affine first factor and nonzero constant step are required");
    const factors=[1n,2n,3n].map(k=>__conventionalAdd(a,[mul(k,d)]));
    return __conventionalDerivative(__conventionalProduct(factors),1);
  }`,
  14: String.raw`function(p,base){return __conventionalMul(poly(p),__conventionalAdd(poly(base),[neg(ONE)]));}`,
  15: String.raw`function(factors){return __conventionalEvaluate(__conventionalProduct(factors),ONE);}`,
  16: String.raw`function(factors,order=1){
    if(!Array.isArray(factors)||!factors.length)throw new RangeError("At least one factor is required");
    if(!Number.isSafeInteger(order)||order<0)throw new RangeError("Derivative order must be nonnegative");
    return __conventionalDerivative(__conventionalProduct(factors),order);
  }`,
  17: String.raw`function(a,b,referenceBase,ratio){
    a=poly(a);b=poly(b);poly(referenceBase);Q(ratio);
    return __conventionalMul(a,b);
  }`,
  18: String.raw`function(dividend,factor){return __conventionalKnownLinear(dividend,factor);}`,
  19: String.raw`function(dividend,factor){return __conventionalKnownLinear(dividend,factor);}`,
  20: String.raw`function(numerator){
    const n=__conventionalInteger(numerator);
    if(n<1n||n>6n)throw new RangeError("A proper positive seventh needs numerator one through six");
    return __conventionalRecurring(n,7n);
  }`,
  21: String.raw`function(p,x){return __conventionalEvaluate(p,x);}`,
  22: String.raw`function(p,base){p=poly(p);poly(base);return __conventionalPower(p,3);}`,
  23: String.raw`function(p,base){p=poly(p);poly(base);return __conventionalPower(p,2);}`,
  24: String.raw`function(prefix,left,right,base){
    prefix=poly(prefix);left=poly(left);right=poly(right);base=poly(base);
    const sum=__conventionalAdd(left,right);
    if(sum.length!==base.length||!sum.every((c,i)=>eq(c,base[i])))
      throw new RangeError("Last portions must sum to the base");
    const start=__conventionalMul(prefix,base);
    return __conventionalMul(__conventionalAdd(start,left),__conventionalAdd(start,right));
  }`,
  25: String.raw`function(first,step){
    first=poly(first);step=Q(step);
    if(first.length!==2||!eq(first[1],ONE)||step.n===0n)
      throw new RangeError("Three monic factors in a nonconstant arithmetic progression are required");
    const factors=[0n,1n,2n].map(k=>__conventionalAdd(first,[mul(k,step)]));
    return __conventionalDerivative(__conventionalProduct(factors),2);
  }`,
  26: String.raw`function(p){return __conventionalEvaluate(p,ONE);}`,
  27: String.raw`function(coefficients,coordinates){
    if(!Array.isArray(coefficients)||coefficients.length!==6||!Array.isArray(coordinates)||coordinates.length!==3)
      throw new RangeError("Six quadratic coefficients and three coordinates are required");
    const c=coefficients.map(x=>Q(x)),[x,y,z]=coordinates.map(v=>Q(v));
    const monomials=[mul(x,x),mul(y,y),mul(z,z),mul(x,y),mul(y,z),mul(z,x)];
    return c.reduce((sum,coefficient,i)=>add(sum,mul(coefficient,monomials[i])),ZERO);
  }`,
  28: String.raw`function(p){
    p=poly(p);
    if((p.length-1)%2!==0||p[0].n===0n)
      throw new RangeError("A reciprocal polynomial needs nonzero even degree endpoints");
    for(let i=0;i<p.length;i++)if(!eq(p[i],p[p.length-1-i]))
      throw new RangeError("Polynomial coefficients are not reciprocal");
    const m=(p.length-1)/2,remainder=Array.from(p),answer=Array(m+1).fill(ZERO);
    for(let k=m;k>=0;k--){
      const coefficient=remainder[m+k];answer[k]=coefficient;
      for(let j=0;j<=k;j++){
        const degree=m+k-2*j;
        remainder[degree]=sub(remainder[degree],mul(coefficient,__conventionalChoose(k,j)));
      }
    }
    if(remainder.some(c=>c.n!==0n))throw new Error("Conventional reciprocal change of variable left a residual");
    return poly(answer);
  }`,
  29: String.raw`function(factors,unknownIndex){
    if(!Array.isArray(factors)||!factors.length)throw new RangeError("At least one factor is required");
    factors=factors.map(poly);
    const left=factors.length===1?poly([ONE]):__conventionalProduct(factors.slice(0,-1));
    const right=factors[factors.length-1],size=left.length+right.length-1;
    if(!Number.isSafeInteger(unknownIndex)||unknownIndex<0||unknownIndex>=size)
      throw new RangeError("The missing coefficient index is outside the product");
    return __conventionalMul(left,right);
  }`
});

function overrideSource(ids) {
  if (!Array.isArray(ids)) throw new TypeError("Classical method ids must be an array");
  const unique = new Set();
  for (const id of ids) {
    if (!Number.isInteger(id) || id < 1 || id > 29)
      throw new RangeError("Classical method ids must be integers from one through twenty-nine");
    if (unique.has(id)) throw new RangeError("Classical method ids must not repeat");
    unique.add(id);
  }
  if (!ids.length) return "";
  let source = helpers;
  for (const id of ids) {
    const name = names[id - 1];
    const body = bodies[id].replace("{", `{__record(${JSON.stringify(name)});`);
    source += `\n${name} = ${body};\n`;
  }
  return source;
}

module.exports = { overrideSource, names };
