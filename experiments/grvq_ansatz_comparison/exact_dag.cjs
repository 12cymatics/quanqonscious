"use strict";

/* An exact rational expression is a value, not an interval-valued state.
 * Outward dyadic enclosures certify decisions only. Every arithmetic operation
 * remains in the exported graph. An unresolved decision raises explicitly.
 */
const crypto = require("node:crypto");
const sha = text => crypto.createHash("sha256").update(text).digest("hex");
const absZ = x => x < 0n ? -x : x;
const bits = x => absZ(x).toString(2).length;
function gcd(a,b) { a=absZ(a);b=absZ(b);while(b)[a,b]=[b,a%b];return a; }
function rat(n,d=1n) {
  if(typeof n!=="bigint"||typeof d!=="bigint")throw new TypeError("Exact rationals require BigInts");
  if(!d)throw new RangeError("Division by zero");
  if(d<0n){n=-n;d=-d;}const g=gcd(n,d);return {n:n/g,d:d/g};
}
function addR(a,b){const g=gcd(a.d,b.d),ad=a.d/g,bd=b.d/g;return rat(a.n*bd+b.n*ad,ad*b.d);}
function mulR(a,b){const g=gcd(a.n,b.d),h=gcd(b.n,a.d);return {n:(a.n/g)*(b.n/h),d:(a.d/h)*(b.d/g)};}
const floorDiv=(n,d)=>{if(d<0n){n=-n;d=-d;}const q=n/d;return n<0n&&n%d?q-1n:q;};
const ceilDiv=(n,d)=>-floorDiv(-n,d);

function createExactDag({precisionBits=128,maxPrecisionBits=65536,leafBits=4096}={}){
  for(const [name,value] of Object.entries({precisionBits,maxPrecisionBits,leafBits}))
    if(!Number.isInteger(value)||value<32)throw new RangeError(`${name} must be an integer at least 32`);
  if(maxPrecisionBits<precisionBits)throw new RangeError("Decision precision ceiling is below starting precision");
  const nodes=[],values=[],interned=new Map(),boundsCaches=new Map(),hashes=[];
  const certificates=[],certificateKeys=new Set(),importedProofSources=new Map();
  const counters={arithmetic_operations:0,exact_leaf_operations:0,structural_identity_decisions:0,
    certified_decisions:0,maximum_certificate_bits:0};
  function recordCertificate(c){const key=JSON.stringify(c);if(!certificateKeys.has(key)){certificateKeys.add(key);certificates.push(c);}}
  function intern(node,key){
    let id=interned.get(key);if(id!==undefined)return values[id];
    id=nodes.length;nodes.push(node);interned.set(key,id);
    values.push(Object.freeze(node.op==="q"?{_expr:id,n:node.n,d:node.d}:{_expr:id}));return values[id];
  }
  function literal(n,d=1n){const r=rat(n,d);return intern({op:"q",...r},`q:${r.n}/${r.d}`);}
  function adopt(x){
    if(x&&Number.isInteger(x._expr)&&values[x._expr]===x)return x;
    if(x&&typeof x.n==="bigint"&&typeof x.d==="bigint")return literal(x.n,x.d);
    throw new TypeError("Exact graph value required; import foreign expression references explicitly");
  }
  const ZERO=literal(0n),ONE=literal(1n),TWO=literal(2n),FOUR=literal(4n);
  const leaf=x=>nodes[x._expr].op==="q";
  const feasible=(a,b)=>bits(a.n)+bits(a.d)+bits(b.n)+bits(b.d)<=leafBits;
  function linearSum(entries){
    const terms=new Map();
    function take(value,scale){
      if(!scale)return;const x=adopt(value),node=nodes[x._expr];
      if(node.op==="q"&&!node.n)return;
      if(node.op==="add"){for(const [id,c]of node.terms)take(values[id],c*scale);return;}
      const updated=(terms.get(x._expr)||0n)+scale;
      if(updated)terms.set(x._expr,updated);else terms.delete(x._expr);
    }
    for(const [value,scale]of entries)take(value,scale);
    if(!terms.size)return ZERO;
    const ordered=[...terms.entries()].sort((a,b)=>a[0]-b[0]);
    if(ordered.length===1&&ordered[0][1]===1n)return values[ordered[0][0]];
    return intern({op:"add",terms:ordered},"a:"+ordered.map(([i,c])=>`${i}:${c}`).join(","));
  }
  function add(a,b){
    counters.arithmetic_operations++;a=adopt(a);b=adopt(b);
    if(leaf(a)&&leaf(b)&&feasible(a,b)){counters.exact_leaf_operations++;const r=addR(a,b);return literal(r.n,r.d);}
    return linearSum([[a,1n],[b,1n]]);
  }
  function neg(a){a=adopt(a);return leaf(a)?literal(-a.n,a.d):linearSum([[a,-1n]]);}
  function sub(a,b){a=adopt(a);b=adopt(b);return a===b?ZERO:add(a,neg(b));}
  function mul(a,b){
    counters.arithmetic_operations++;a=adopt(a);b=adopt(b);
    if(a===ZERO||b===ZERO)return ZERO;if(a===ONE)return b;if(b===ONE)return a;
    if(leaf(a)&&leaf(b)&&feasible(a,b)){counters.exact_leaf_operations++;const r=mulR(a,b);return literal(r.n,r.d);}
    if(leaf(a)&&a.n===-a.d)return neg(b);if(leaf(b)&&b.n===-b.d)return neg(a);
    const i=Math.min(a._expr,b._expr),j=Math.max(a._expr,b._expr);
    return intern({op:"mul",a:i,b:j},`m:${i},${j}`);
  }
  function linearMap(N,D,psi){
    if(typeof D!=="bigint"||D<=0n||!Array.isArray(psi))throw new TypeError("Invalid exact matrix denominator/state");
    const inputs=psi.map(adopt);
    return N.map(row=>{
      if(row.length!==inputs.length||row.some(n=>typeof n!=="bigint"))throw new TypeError("Invalid exact matrix row");
      const terms=row.map((n,k)=>[inputs[k]._expr,n]).filter(x=>x[1]);
      if(!terms.length)return ZERO;
      return intern({op:"lin",d:D,terms},`l:${D}:`+terms.map(([id,n])=>`${id}:${n}`).join(","));
    });
  }
  const dependencies=n=>n.op==="q"?[]:n.op==="add"||n.op==="lin"?n.terms.map(t=>t[0]):[n.a,n.b];
  function enclosure(value,precision=precisionBits){
    value=adopt(value);if(!Number.isInteger(precision)||precision<1)throw new RangeError("Invalid certificate precision");
    let cache=boundsCaches.get(precision);if(!cache){cache=new Map();boundsCaches.set(precision,cache);}
    const scale=1n<<BigInt(precision),stack=[[value._expr,false]];
    while(stack.length){
      const [id,expanded]=stack.pop();if(cache.has(id))continue;
      const node=nodes[id],deps=dependencies(node);
      if(!expanded&&deps.some(i=>!cache.has(i))){stack.push([id,true]);for(const i of deps)if(!cache.has(i))stack.push([i,false]);continue;}
      let lo,hi;
      if(node.op==="q"){lo=floorDiv(node.n*scale,node.d);hi=ceilDiv(node.n*scale,node.d);}
      else if(node.op==="add"||node.op==="lin"){
        lo=0n;hi=0n;for(const [i,c]of node.terms){const x=cache.get(i);lo+=c*(c>0n?x.lo:x.hi);hi+=c*(c>0n?x.hi:x.lo);}
        if(node.op==="lin"){lo=floorDiv(lo,node.d);hi=ceilDiv(hi,node.d);}
      }else{
        const a=cache.get(node.a),b=cache.get(node.b),corners=[[a.lo,b.lo],[a.lo,b.hi],[a.hi,b.lo],[a.hi,b.hi]];
        if(node.op==="mul"){
          const p=corners.map(([x,y])=>x*y);lo=floorDiv(p.reduce((x,y)=>x<y?x:y),scale);hi=ceilDiv(p.reduce((x,y)=>x>y?x:y),scale);
        }else if(node.op==="div"){
          if(b.lo<=0n&&b.hi>=0n){const error=new Error("A divisor enclosure contains zero");error.needsPrecision=true;throw error;}
          lo=corners.map(([x,y])=>floorDiv(x*scale,y)).reduce((x,y)=>x<y?x:y);
          hi=corners.map(([x,y])=>ceilDiv(x*scale,y)).reduce((x,y)=>x>y?x:y);
        }else throw new Error(`Unknown exact operation ${node.op}`);
      }
      cache.set(id,{lo,hi,bits:precision});
    }
    return cache.get(value._expr);
  }
  function exactLinearZero(value){
    const memo=new Map();let visits=0;
    function form(id){
      if(memo.has(id))return memo.get(id);if(++visits>20000)return null;
      const node=nodes[id];let result=new Map();
      function append(part,coefficient){
        if(!part)return false;
        for(const [key,c]of part){const next=addR(result.get(key)||{n:0n,d:1n},mulR(c,coefficient));if(next.n)result.set(key,next);else result.delete(key);}
        return true;
      }
      if(node.op==="q"){if(node.n)result.set(-1,{n:node.n,d:node.d});}
      else if(node.op==="add"||node.op==="lin"){
        for(const [child,c]of node.terms)if(!append(form(child),{n:c,d:node.op==="lin"?node.d:1n}))return null;
      }else if(node.op==="mul"&&(nodes[node.a].op==="q"||nodes[node.b].op==="q")){
        const constant=nodes[node.a].op==="q"?node.a:node.b;
        if(!append(form(constant===node.a?node.b:node.a),nodes[constant]))return null;
      }else if(node.op==="div"&&nodes[node.b].op==="q"){
        if(!append(form(node.a),rat(nodes[node.b].d,nodes[node.b].n)))return null;
      }else result.set(id,{n:1n,d:1n});
      memo.set(id,result);return result;
    }
    const result=form(value._expr);return result!==null&&result.size===0;
  }
  function certify(value,kind){
    value=adopt(value);
    if(leaf(value)){counters.structural_identity_decisions++;return kind==="floor"?floorDiv(value.n,value.d):value.n<0n?-1:value.n>0n?1:0;}
    let attemptedIdentity=false;
    for(let precision=precisionBits;precision<=maxPrecisionBits;precision*=2){
      let bounds;try{bounds=enclosure(value,precision);}catch(error){if(error.needsPrecision)continue;throw error;}
      let result;
      if(kind==="floor"){
        const scale=1n<<BigInt(precision),a=floorDiv(bounds.lo,scale),b=floorDiv(bounds.hi,scale);if(a===b)result=a;
      }else if(bounds.lo>0n)result=1;else if(bounds.hi<0n)result=-1;else if(bounds.lo===0n&&bounds.hi===0n)result=0;
      if(result!==undefined){
        recordCertificate({expression:value._expr,kind,bits:precision,lower_integer:bounds.lo.toString(),upper_integer:bounds.hi.toString(),result:result.toString()});
        counters.certified_decisions++;counters.maximum_certificate_bits=Math.max(counters.maximum_certificate_bits,precision);return result;
      }
      if(!attemptedIdentity){
        attemptedIdentity=true;
        if(exactLinearZero(value)){recordCertificate({expression:value._expr,kind,proof:"exact_rational_linear_identity",result:"0"});counters.structural_identity_decisions++;return kind==="floor"?0n:0;}
        if(kind==="floor"){
          const scale=1n<<BigInt(precision),candidate=ceilDiv(bounds.lo,scale);
          if(candidate*scale<=bounds.hi&&exactLinearZero(sub(value,literal(candidate)))){
            recordCertificate({expression:value._expr,kind,proof:"exact_rational_linear_integer",result:candidate.toString()});counters.structural_identity_decisions++;return candidate;
          }
        }
      }
    }
    const error=new Error(`Cannot certify ${kind} for expression ${value._expr} within explicit ${maxPrecisionBits}-bit ceiling; no value was substituted`);
    error.name="ExactDecisionUnresolved";error.expression=value._expr;throw error;
  }
  function div(a,b){
    counters.arithmetic_operations++;a=adopt(a);b=adopt(b);
    if(certify(b,"sign")===0)throw new RangeError("Division by zero");
    if(a===ZERO)return ZERO;if(a===b)return ONE;if(b===ONE)return a;
    if(leaf(a)&&leaf(b)&&feasible(a,b)){counters.exact_leaf_operations++;const r=mulR(a,rat(b.d,b.n));return literal(r.n,r.d);}
    return intern({op:"div",a:a._expr,b:b._expr},`d:${a._expr},${b._expr}`);
  }
  function digest(value){
    value=adopt(value);
    for(let id=0;id<=value._expr;id++)if(!hashes[id]){
      const n=nodes[id];
      const representation=n.op==="q"?["q",n.n.toString(),n.d.toString()]
        :n.op==="add"||n.op==="lin"?[n.op,n.d?.toString()||"1",n.terms.map(([i,c])=>[hashes[i],c.toString()]).sort((a,b)=>a[0].localeCompare(b[0]))]
        :[n.op,...(n.op==="mul"?[hashes[n.a],hashes[n.b]].sort():[hashes[n.a],hashes[n.b]])];
      hashes[id]=sha(JSON.stringify(representation));
    }return hashes[value._expr];
  }
  function encode(value){
    if(value&&typeof value==="object"&&Number.isInteger(value._expr))return {_expr:adopt(value)._expr};
    if(value&&typeof value.n==="bigint"&&typeof value.d==="bigint")return {n:value.n.toString(),d:value.d.toString()};
    if(typeof value==="bigint")return value.toString();
    if(value===null||value===undefined||["string","number","boolean"].includes(typeof value))return value;
    if(Array.isArray(value))return value.map(encode);
    if(typeof value==="object")return Object.fromEntries(Object.entries(value).filter(([,v])=>typeof v!=="function").map(([k,v])=>[k,encode(v)]));
    throw new TypeError("Unsupported graph serialization value");
  }
  function decodeImported(value,idMap=values){
    if(value&&typeof value==="object"&&Object.keys(value).length===1&&Number.isInteger(value._expr)){
      if(!idMap[value._expr])throw new RangeError("Unknown expression reference");return idMap[value._expr];
    }
    if(value&&typeof value==="object"&&Object.keys(value).length===2&&typeof value.n==="string"&&typeof value.d==="string")return literal(BigInt(value.n),BigInt(value.d));
    if(Array.isArray(value))return value.map(v=>decodeImported(v,idMap));
    if(value&&typeof value==="object")return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,decodeImported(v,idMap)]));
    return value;
  }
  function stats(){return {...counters,nodes:nodes.length,retained_certificates:certificates.length,interval_values_in_state:0,
    rational_leaf_bit_budget:leafBits,certificate_start_bits:precisionBits,certificate_max_bits:maxPrecisionBits};}
  function exportGraph(){return {schema:"exact-rational-expression-dag/v1",settings:{precisionBits,maxPrecisionBits,leafBits},
    nodes:nodes.map(n=>n.op==="q"?["q",n.n.toString(),n.d.toString()]:n.op==="add"?["add",n.terms.map(([i,c])=>[i,c.toString()])]
      :n.op==="lin"?["lin",n.d.toString(),n.terms.map(([i,c])=>[i,c.toString()])]:[n.op,n.a,n.b]),
    certificates:certificates.slice(),imported_proof_sources:[...importedProofSources.values()],statistics:stats()};}
  function importGraph(graph){
    if(graph?.schema!=="exact-rational-expression-dag/v1"||!Array.isArray(graph.nodes))throw new TypeError("Invalid expression graph");
    const map=[];
    for(let index=0;index<graph.nodes.length;index++){
      const n=graph.nodes[index],get=i=>{if(!Number.isInteger(i)||i<0||i>=index)throw new RangeError("Graph must be topologically ordered");return map[i];};
      let value;
      if(n[0]==="q")value=literal(BigInt(n[1]),BigInt(n[2]));
      else if(n[0]==="add")value=linearSum(n[1].map(([i,c])=>[get(i),BigInt(c)]));
      else if(n[0]==="lin")value=linearMap([n[2].map(t=>BigInt(t[1]))],BigInt(n[1]),n[2].map(t=>get(t[0])))[0];
      else if(n[0]==="mul")value=mul(get(n[1]),get(n[2]));
      else if(n[0]==="div")value=div(get(n[1]),get(n[2]));
      else throw new RangeError("Unknown imported graph operation");map.push(value);
    }
    if(graph.certificates!==undefined&&!Array.isArray(graph.certificates))throw new TypeError("Invalid certificate ledger");
    const sourceHash=sha(JSON.stringify({nodes:graph.nodes,certificates:graph.certificates||[]}));
    for(const certificate of graph.certificates||[]){
      if(!Number.isInteger(certificate.expression)||!map[certificate.expression])throw new RangeError("Certificate references missing expression");
      recordCertificate({...certificate,expression:map[certificate.expression]._expr});
    }
    importedProofSources.set(sourceHash,{source_graph_and_certificates_sha256:sourceHash,imported_nodes:map.length,imported_certificates:(graph.certificates||[]).length});
    return map;
  }
  const Q=Object.freeze({
    mk:literal,parse(text){const s=String(text).trim();if(!/^[+-]?\d+(?:\/[+-]?\d+)?$/.test(s))throw new SyntaxError("Invalid exact rational literal");const p=s.split("/");return literal(BigInt(p[0]),p[1]===undefined?1n:BigInt(p[1]));},
    add,sub,mul,div,neg,abs(x){x=adopt(x);return certify(x,"sign")<0?neg(x):x;},
    cmp(a,b){return certify(sub(a,b),"sign");},sign(x){return certify(x,"sign");},eq(a,b){return certify(sub(a,b),"sign")===0;},
    isZero(x){return certify(x,"sign")===0;},agree(a,b){return certify(sub(a,b),"sign")===0;},
    mayLt(a,b){return certify(sub(a,b),"sign")<0;},mayLe(a,b){return certify(sub(a,b),"sign")<=0;},mayZero(x){return certify(x,"sign")===0;},
    min(a,b){a=adopt(a);b=adopt(b);return certify(sub(a,b),"sign")<=0?a:b;},max(a,b){a=adopt(a);b=adopt(b);return certify(sub(a,b),"sign")>=0?a:b;},
    floor(x){return certify(x,"floor");},pow(x,k){if(!Number.isInteger(k)||k<0)throw new RangeError("Invalid integer exponent");let out=ONE,b=adopt(x);while(k){if(k%2)out=mul(out,b);k=Math.floor(k/2);if(k)b=mul(b,b);}return out;},
    isExact(x){try{adopt(x);return true;}catch{return false;}},isValue(x){try{adopt(x);return true;}catch{return false;}},
    lo:adopt,hi:adopt,width(){return ZERO;},bits(x){x=adopt(x);return leaf(x)?{numeratorBits:bits(x.n),denominatorBits:bits(x.d)}:{numeratorBits:null,denominatorBits:null,exact_expression:x._expr};},
    toString(x){x=adopt(x);return leaf(x)?x.d===1n?x.n.toString():`${x.n}/${x.d}`:`exact_expr#${x._expr}`;},
    ZERO,ONE,TWO,FOUR,HALF:literal(1n,2n),QUARTER:literal(1n,4n),SIXTEENTH:literal(1n,16n)
  });
  return {Q,encode,decode:value=>decodeImported(value),exportGraph,importGraph,decodeImported,
    lookup(id){if(!Number.isInteger(id)||!values[id])throw new RangeError("Unknown expression id");return values[id];},
    digest,linearMap,enclosure,stats,certificates};
}
module.exports={createExactDag};
