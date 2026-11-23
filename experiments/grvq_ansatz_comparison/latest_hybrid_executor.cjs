"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const crypto = require("node:crypto");
const zlib = require("node:zlib");
const { performance } = require("node:perf_hooks");
const { Worker, isMainThread, parentPort, workerData, threadId } = require("node:worker_threads");

const ROOT = path.resolve(__dirname, "../..");
const SOURCE_PATH = path.join(ROOT, "vedic_v18.51.1_exact_phi.html");
const SOURCE = fs.readFileSync(SOURCE_PATH, "utf8");
const BEGIN = "const KERNEL_FACTORY = () => {", END = "const KERNEL = KERNEL_FACTORY();";
const first = SOURCE.indexOf(BEGIN), last = SOURCE.indexOf(END, first);
if (first < 0 || last <= first || SOURCE.indexOf(BEGIN, first + BEGIN.length) !== -1)
  throw new Error("Exactly one source KERNEL_FACTORY must exist");
const FACTORY_SOURCE = SOURCE.slice(first, last).trim();
const hash = value => crypto.createHash("sha256").update(value).digest("hex");
const SOURCE_INFO = Object.freeze({
  path: path.relative(ROOT, SOURCE_PATH),
  source_commit: "13ec5d881b1447a8bf9de87a1f0dec1773dcef3c",
  declared_version: /const VERSION = Object\.freeze\(\{\s*string:\s*'([^']+)'/.exec(SOURCE)?.[1],
  sha256: hash(SOURCE), kernel_factory_sha256: hash(FACTORY_SOURCE),
  scope: "Source formulaOf: BOOK transforms, variable GRVQ, and MSTVQ stress; not the additional browser stagesFrom evolution",
});
const COMPOSITIONS = Object.freeze(["SERIES", "PARALLEL", "SYMMETRIC_CONCURRENT"]);

function replaceOnce(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("Source representation boundary changed: " + before.slice(0,70));
  return source.replace(before, after);
}

// Every source matching is a complete, consistently oriented hypercube axis.
// Its eight matchings repeat each of the four axes twice. Rotations on
// different axes commute, and rotations on one axis are SO(2) factors.
// Consequently the *same integer matrix*, including its unreduced denominator,
// is the fourfold tensor power of the product of all local Gaussian factors.
// This changes only multiplication order: every returned coefficient retains
// its exact multiplicity, including repeated repetend digits.
function createTensorMatrix(matchings, R, getQ) {
  if (!Array.isArray(matchings) || matchings.length !== 8) throw new Error("The source must have eight proven hypercube matchings");
  const axisCounts = [0,0,0,0], axes = [];
  for (const matching of matchings) {
    if (matching.length !== 8) throw new Error("A source matching must cover all sixteen vertices");
    const seen = new Set(); let axis = null;
    for (const [l,r] of matching) {
      const difference = l ^ r, bit = [1,2,4,8].indexOf(difference);
      if (bit < 0 || l >= r || l < 0 || r > 15 || seen.has(l) || seen.has(r) || (axis !== null && axis !== bit))
        throw new Error("The source matching no longer proves the tensor identity");
      axis = bit; seen.add(l); seen.add(r);
    }
    if (seen.size !== 16) throw new Error("The source matching is incomplete");
    axes.push(axis); axisCounts[axis]++;
  }
  if (axisCounts.some(n=>n!==2)) throw new Error("Each source hypercube axis must occur exactly twice");
  const gcd = (a,b)=>{a=a<0n?-a:a;b=b<0n?-b:b;while(b){const r=a%b;a=b;b=r;}return a;};
  const multiply = (a,b)=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0],a[2]*b[2]];
  const power = (factor,n)=>{let result=[1n,0n,1n],base=factor;while(n>0){if(n%2===1)result=multiply(result,base);n=Math.floor(n/2);if(n)base=multiply(base,base);}return result;};
  const popcount = n=>{let count=0;while(n){count+=n&1;n>>>=1;}return count;};
  const cache = new Map();
  return (coefficients,strength)=>{
    if(!getQ().isExact(strength))throw new TypeError("The strength of a book rotation is an exact rational");
    const key=`${strength.n}/${strength.d}|${coefficients.map(c=>`${c.n}/${c.d}`).join(',')}`;
    const kept=cache.get(key);if(kept)return kept;
    const groups=new Map();let zeroTerms=0,visited=0;
    for(const coefficient of coefficients){
      visited++;
      const t=R.mul(strength,coefficient),p=t.n,q=t.d;
      if(p===0n){zeroTerms++;continue;}
      let base=q*q+p*p,cos=q*q-p*p,sin=2n*p*q;
      const g=gcd(gcd(base,cos),sin);base/=g;cos/=g;sin/=g;
      const factorKey=`${cos},${sin},${base}`,group=groups.get(factorKey);
      if(group)group.count++;else groups.set(factorKey,{factor:[cos,sin,base],count:1});
    }
    let products=Array.from(groups.values(),group=>power(group.factor,2*group.count));
    if(products.length===0)products=[[1n,0n,1n]];
    while(products.length>1){const next=[];for(let i=0;i<products.length;i+=2)next.push(i+1<products.length?multiply(products[i],products[i+1]):products[i]);products=next;}
    const [C,S,Dsingle]=products[0];
    if(C*C+S*S!==Dsingle*Dsingle)throw new Error("Exact Gaussian product failed its norm identity");
    const cp=[1n,C],sp=[1n,S];for(let i=2;i<=4;i++){cp.push(cp[i-1]*C);sp.push(sp[i-1]*S);}
    const magnitudes=Array.from({length:5},(_,h)=>cp[4-h]*sp[h]);
    const N=Array.from({length:16},(_,i)=>Object.freeze(Array.from({length:16},(_,j)=>{
      const n=magnitudes[popcount(i^j)];return popcount((~i)&j&15)%2?-n:n;
    })));
    const out=Object.freeze({N:Object.freeze(N),D:Dsingle**4n,tensor_certificate:Object.freeze({
      representation:"exact_commuting_gaussian_tensor",matching_axes:axes,axis_multiplicities:axisCounts,
      visited_terms:visited,zero_identity_terms:zeroTerms,nonzero_terms:visited-zeroTerms,
      total_source_matching_rotations:8*(visited-zeroTerms),unique_gaussian_factors:groups.size,
      grouped_multiplicities:Array.from(groups.values(),g=>g.count),single_axis_numerator:[C,S],single_axis_denominator:Dsingle,
      exact_norm_identity:true,unreduced_source_denominator_preserved:true})});
    if(cache.size>=64)cache.clear();cache.set(key,out);return out;
  };
}

function loadKernel(options = {}) {
  const arithmetic = options.arithmetic || "expanded_rational";
  if (!["expanded_rational", "exact_dag"].includes(arithmetic)) throw new RangeError("Select expanded_rational or exact_dag explicitly");
  const dagOptions = options.dagOptions || {};
  const matrixRepresentation = options.matrixRepresentation || "commuting_tensor";
  if(!["commuting_tensor","source_dense"].includes(matrixRepresentation))throw new RangeError("Select the exact tensor representation or the explicit original source oracle");
  const arena = arithmetic === "exact_dag" ? require("./exact_dag.cjs").createExactDag(dagOptions) : null;
  let source = FACTORY_SOURCE;
  source = replaceOnce(source,
    "      const period = d => d <= 999n ? d : refuse(`the repetend of 1/${d} can run to ${d - 1n} digits; above three digits it is refused`);",
    "      const period = d => d > 1n && gcdZ(d, 10n) === 1n ? d : refuse('a repetend denominator must exceed one and be coprime to ten');");
  source = replaceOnce(source, "return Object.freeze({sourcesData,",
    "return Object.freeze({makeKeep,narrow,TGCR_KAPPA_FEED," + (arena ? "setArithmetic:q=>{Q=q;}," : "") + "sourcesData,");
  const representationEdits = [];
  if(matrixRepresentation === "commuting_tensor"){
    source=replaceOnce(source,"    const matrixOf = (coefficients, strength) => {","    const sourceMatrixOf = (coefficients, strength) => {");
    source=replaceOnce(source,"    const rotate = (psi, coefficients, strength) => {","    const matrixOf = __EXACT_MATRIX(MATCHINGS, R, () => Q);\n    const rotate = (psi, coefficients, strength) => {");
    representationEdits.push("BOOK.matrixOf uses the proven exact Gaussian product and fourfold tensor representation of all eight source matchings, preserving every coefficient multiplicity and the original unreduced N and D");
  }
  if (arena) {
    source = replaceOnce(source, "  const Q = Object.freeze({", "  let Q = Object.freeze({");
    const signStart = source.indexOf("  const sgnQ = x => {"), signEnd = source.indexOf("  const SUTRA_SUM",signStart);
    if (signStart < 0 || signEnd < signStart) throw new Error("Source sign helper boundary changed");
    source = source.slice(0,signStart)+"  const sgnQ = x => Q.sign(x) < 0 ? -1n : 1n;\n"+source.slice(signEnd);
    const rotateStart = source.indexOf("    const rotate = (psi, coefficients, strength) => {"), rotateEnd = source.indexOf("    const apply = (id, psi, strength, operands)",rotateStart);
    if (rotateStart < 0 || rotateEnd < rotateStart) throw new Error("BOOK.rotate boundary changed");
    source = source.slice(0,rotateStart)+
      "    const rotate = (psi, coefficients, strength) => {\n"+
      "      const { N, D } = matrixOf(coefficients, strength);\n"+
      "      return __EXACT_ARENA.linearMap(N, D, psi);\n"+
      "    };\n"+source.slice(rotateEnd);
    representationEdits.push("Q is rebound after original source initialization to exact rational DAG operations",
      "sgnQ obtains its exact sign from Q.sign instead of directly inspecting an expanded numerator",
      "BOOK.rotate applies the identical exact matrixOf(N,D) using exact DAG linearMap");
  }
  const factory = vm.runInNewContext(source+"\nKERNEL_FACTORY;", { __EXACT_ARENA: arena, __EXACT_MATRIX:createTensorMatrix }, { filename: SOURCE_PATH });
  const kernel = factory();
  kernel.setExactBits(null);
  if (arena) kernel.setArithmetic(arena.Q);
  if (kernel.exactBits() !== null) throw new Error("Source unlimited exact arithmetic was not enabled");
  return Object.freeze({ ...kernel, Q: arena ? arena.Q : kernel.Q, exactDag: arena,
    arithmetic, dagOptions, matrix_representation:matrixRepresentation, representation_edits: representationEdits,
    domain_extensions: ["S1/S12 retain their full exact repetend algorithms for every source-generated valid denominator; the prior denominator<=999 workload refusal is removed"],
    arithmetic_sha256: arena ? hash(fs.readFileSync(path.join(__dirname,"exact_dag.cjs"))) : null });
}

function encode(K, value, includeGraph = false) {
  if (!K.exactDag) return K.serializeExact(value);
  const encoded = K.exactDag.encode(value);
  return includeGraph ? { ...encoded, expression_graph: K.exactDag.exportGraph() } : encoded;
}

function parseQ(K, value, name) {
  if (typeof value === "string") return K.Q.parse(value);
  if (value && typeof value === "object" && Object.keys(value).length === 2 &&
      "n" in value && "d" in value && [value.n,value.d].every(v=>typeof v==="string"||typeof v==="bigint"))
    return K.Q.mk(BigInt(value.n),BigInt(value.d));
  throw new TypeError(`${name} must be an explicit exact rational string or {n,d}`);
}

function exactState(K,state,name) {
  if (!Array.isArray(state)||state.length!==16||state.some(q=>!K.Q.isValue(q)||!K.Q.isExact(q)))
    throw new TypeError(`${name} requires sixteen exact rational values`);
  return state;
}
const equalStates = (K,a,b) => a.length===b.length && a.every((v,i)=>K.Q.eq(v,b[i]));
const complexity = (K,state) => K.exactDag ? { representation:"exact_rational_expression_dag",
  ...K.exactDag.stats(), state_root_digests:state.map(q=>K.exactDag.digest(q)), expanded_rational_bit_lengths:null }
  : K.complexity(state);

function prepare(K,request,options) {
  if (!request||typeof request!=="object"||Array.isArray(request)) throw new TypeError("Supply an explicit request object");
  if (!COMPOSITIONS.includes(request.composition)) throw new RangeError("Select an explicit mathematical composition");
  if ((request.arithmetic||"expanded_rational")!==K.arithmetic) throw new RangeError("Request and kernel arithmetic differ");
  if (request.exact_bits!==undefined && request.exact_bits!==null) throw new RangeError("The experiment requires exact_bits=null");
  const backend=request.backend||"serial", workers=request.workers===undefined?2:request.workers;
  if (!["serial","workers"].includes(backend)) throw new RangeError("CPU backend must be serial or workers");
  if (backend==="workers"&&request.composition!=="PARALLEL") throw new RangeError("Within-field workers require independent PARALLEL branches; SERIES and concurrent dependencies must be retained");
  if (!Number.isInteger(workers)||workers<1) throw new RangeError("Worker count must be a positive integer");
  let idMap;
  if (request.expression_graph) {
    if (!K.exactDag) throw new TypeError("An expression graph needs exact_dag arithmetic");
    idMap=K.exactDag.importGraph(request.expression_graph);
  }
  let state;
  if (request.initial_state==="source"&&request.state===undefined) state=K.initState();
  else if (Array.isArray(request.state)&&request.initial_state===undefined) state=options.rawState?request.state:
    request.state.map((q,i)=>idMap?K.exactDag.decodeImported(q,idMap):parseQ(K,q,`state[${i}]`));
  else throw new TypeError("Supply state[16] or select initial_state='source'");
  exactState(K,state,"Initial field");
  const list=request.active===undefined&&request.strength!==undefined?
    Array.from({length:29},(_,i)=>({id:i+1,strength:request.strength})):request.active;
  if (!Array.isArray(list)||!list.length) throw new TypeError("Supply active methods and nonzero strengths");
  const active=list.map(s=>{
    if(!s||!Number.isInteger(s.id)||s.id<1||s.id>29)throw new RangeError("Method IDs range from 1 through 29");
    const a={id:s.id};
    if(s.args!==undefined){if(!Array.isArray(s.args))throw new TypeError("Method args must be an exact array");a.args=s.args.map((q,i)=>parseQ(K,q,`S${s.id}.args[${i}]`));K.sutraArgs(s.id,a.args);}
    else {a.strength=parseQ(K,s.strength,`S${s.id}.strength`);K.checkStrength(s.id,a.strength);}
    if(s.gain!==undefined)a.gain=parseQ(K,s.gain,`S${s.id}.gain`);
    return a;
  }).sort((a,b)=>a.id-b.id);
  if(new Set(active.map(a=>a.id)).size!==active.length)throw new RangeError("Methods must not be repeated in the activation list");
  if(request.require_all_29!==false){
    if(active.length!==29||active.some((s,i)=>s.id!==i+1))throw new RangeError("The complete trajectory requires all twenty-nine methods");
    if(active.some(s=>K.Q.isZero((s.args||[s.strength])[0])||(s.gain&&K.Q.isZero(s.gain))))throw new RangeError("All twenty-nine strengths and gains must be nonzero");
  }
  const time=options.rawState&&K.Q.isValue(request.time)?request.time:parseQ(K,request.time===undefined?"0":request.time,"time");
  if(K.Q.sign(time)<0)throw new RangeError("Source time must be nonnegative");
  return {state,active,time,backend,workers,composition:request.composition};
}

class WorkerPool {
  constructor(size,K){
    this.K=K;this.queue=[];this.slots=[];this.closed=false;
    for(let i=0;i<size;i++){
      const worker=new Worker(__filename,{workerData:{latestHybridWorker:true,arithmetic:K.arithmetic,dagOptions:K.dagOptions,matrixRepresentation:K.matrix_representation}});
      const slot={worker,ready:false,job:null};this.slots.push(slot);
      worker.on("message",message=>{
        if(message.ready){slot.ready=true;this.dispatch();return;}
        const job=slot.job;slot.job=null;if(!job)return;
        if(message.error){const error=new Error(message.error.message);error.name=message.error.name;error.stack=message.error.stack;job.reject(error);}
        else if(message.expression_graph){try{const ids=this.K.exactDag.importGraph(message.expression_graph);job.resolve(this.K.exactDag.decodeImported(message.value,ids));}catch(error){job.reject(error);}}
        else job.resolve(message.value);
        this.dispatch();
      });
      worker.on("error",error=>this.fail(error));
      worker.on("exit",code=>{if(!this.closed&&code!==0)this.fail(new Error(`Exact worker exited with code ${code}`));});
    }
  }
  submit(payload){
    if(this.closed)return Promise.reject(new Error("Worker pool is closed"));
    if(payload.kind==="branch"&&this.K.exactDag)payload={kind:"branch",encoded:this.K.exactDag.encode(payload),expression_graph:this.K.exactDag.exportGraph()};
    return new Promise((resolve,reject)=>{this.queue.push({payload,resolve,reject});this.dispatch();});
  }
  dispatch(){if(this.closed)return;for(const slot of this.slots)if(slot.ready&&!slot.job&&this.queue.length){slot.job=this.queue.shift();slot.worker.postMessage(slot.job.payload);}}
  fail(error){for(const job of this.queue.splice(0))job.reject(error);for(const slot of this.slots)if(slot.job){slot.job.reject(error);slot.job=null;}}
  async close(){this.closed=true;await Promise.all(this.slots.map(s=>s.worker.terminate()));}
}

function branch(K,state,sutra,time){
  const records=[],start=performance.now();
  const output=K.applySutra(sutra.id,state,sutra.args===undefined?sutra.strength:sutra.args,
    {t:time,gain:sutra.gain||K.Q.ONE,record:records});
  exactState(K,output,`S${sutra.id} output`);
  if(records.length!==1)throw new Error("Every source method must emit one execution record");
  return {output,record:records[0],worker_thread_id:threadId,elapsed_ms:performance.now()-start};
}

async function execute(request,options={}){
  const started=performance.now(),K=options.kernel||loadKernel({arithmetic:request.arithmetic,dagOptions:request.dag_options});
  const cfg=prepare(K,request,options),run=K.makeKeep();
  const records=[],grvqRecords=[],workerReceipts=[];
  const progress=options.onProgress||(()=>{}),journal=options.onJournal||(()=>{});
  let previous=cfg.state;
  journal({stage:"source_formula_input",input:cfg.state,active:cfg.active,time:cfg.time,composition:cfg.composition});
  const methodRecord=raw=>{
    const before=cfg.composition==="PARALLEL"?cfg.state:previous;
    const after=raw.grvq?K.injectLambda(raw.info.p,raw.grvq.k,raw.grvq.delta):raw.info.p;
    exactState(K,after,`S${raw.id} coupled output`);
    const grvq=raw.grvq?{...raw.grvq,terms:K.grvqTerms(raw.grvq.lambda,raw.grvq.k,cfg.time)}:null;
    if(grvq){const sum=Object.values(grvq.terms).reduce(K.Q.add,K.Q.ZERO);if(!K.Q.eq(K.Q.mul(sum,raw.alpha),grvq.delta))throw new Error("GRVQ term sum differs from the consumed delta");}
    const record={sequence:records.length+1,id:raw.id,type:K.SUTRA_TYPE[raw.id],before,after_book_rotation:raw.info.p,after,
      alpha:raw.alpha,operands:raw.info.operands,terms:raw.info.terms,
      rotation_parameters:raw.info.terms.map(t=>K.Q.mul(raw.alpha,t)),
      source_zero_term_rule:"Exact zero terms are identity rotations in source BOOK.matrixOf",
      grvq,mstvq_book_field_diagnostic:raw.B,energy_change_exact:raw.dE,
      consumed_by:cfg.composition==="PARALLEL"?"common_input_displacement_sum":"next ordered method or distributed_GRVQ"};
    if(K.matrix_representation==="commuting_tensor"){
      const {single_axis_numerator,single_axis_denominator,...certificate}=K.BOOK.matrixOf(raw.info.terms,raw.alpha).tensor_certificate;
      if(certificate.visited_terms!==raw.info.terms.length)throw new Error("The exact matrix did not consume all source coefficients");
      record.exact_matrix_representation=certificate;
    }
    records.push(record);previous=after;journal({stage:"sutra",record});
    progress({stage:"sutra",sequence:record.sequence,id:raw.id,elapsed_ms:performance.now()-started});
  };
  let afterS;
  if(cfg.backend==="workers"){
    const pool=new WorkerPool(cfg.workers,K);
    try{
      const outputs=await Promise.all(cfg.active.map(sutra=>pool.submit({kind:"branch",state:cfg.state,sutra,time:cfg.time})));
      let displacement=cfg.state.map(()=>K.Q.ZERO);
      for(let i=0;i<outputs.length;i++){
        const output=outputs[i];methodRecord(output.record);
        if(!equalStates(K,records[records.length-1].after,output.output))throw new Error("Worker output differs from its exact receipt");
        const kept=run.keep(output.output);displacement=displacement.map((d,j)=>K.Q.add(d,K.Q.sub(kept[j],cfg.state[j])));
        workerReceipts.push({id:cfg.active[i].id,worker_thread_id:output.worker_thread_id,elapsed_ms:output.elapsed_ms});
      }
      afterS=run.keep(cfg.state.map((x,i)=>K.Q.add(x,displacement[i])));
    }finally{await pool.close();}
  }else afterS=K.applySutraCore(cfg.state,cfg.active,cfg.composition,run.keep,{t:cfg.time,record:{push:methodRecord}});
  exactState(K,afterS,"Core output");
  if(cfg.composition!=="PARALLEL"&&!equalStates(K,previous,afterS))throw new Error("Ordered method receipts do not reconstruct the core output");
  previous=afterS;journal({stage:"core_complete",afterS});
  const afterD=K.distributedGRVQ(afterS,cfg.active,cfg.time,run.keep,{push(raw){
    const before=previous,after=K.injectLambda(before,raw.k,raw.delta);
    exactState(K,after,`S${raw.id} distributed GRVQ output`);
    const record={sequence:grvqRecords.length+1,...raw,terms:K.grvqTerms(raw.lambda,raw.k,cfg.time),before,after,
      consumed_by:"next distributed GRVQ update or MSTVQ stress"};
    grvqRecords.push(record);previous=after;journal({stage:"distributed_grvq",record});
    progress({stage:"distributed_grvq",id:raw.id,sequence:record.sequence,elapsed_ms:performance.now()-started});
  }});
  if(!equalStates(K,previous,afterD))throw new Error("Distributed GRVQ receipts do not reconstruct output");
  const mstvq=K.distributedMSTVQ(afterD,cfg.active,run.keep);K.narrow(mstvq.psi,"the width of the formula");
  exactState(K,mstvq.psi,"Final coupled field");
  const expected=cfg.active.length*(cfg.composition==="SYMMETRIC_CONCURRENT"?2:1);
  if(records.length!==expected)throw new Error("The composition did not execute all required method applications");
  if(grvqRecords.length!==cfg.active.filter(s=>K.GRVQ_K[s.id]===undefined).length)throw new Error("A distributed GRVQ contribution is missing");
  const result={schema:"quanqonscious.latest-hybrid-execution.v2",source:SOURCE_INFO,
    case_id:request.case_id||null,mathematical_composition:cfg.composition,cpu_backend:cfg.backend,
    workers:cfg.backend==="workers"?cfg.workers:0,arithmetic:K.arithmetic,arithmetic_module_sha256:K.arithmetic_sha256,
    representation_edits:K.representation_edits,matrix_representation:K.matrix_representation,domain_extensions:K.domain_extensions,exact_bits:null,intervals_used:false,
    composition_equation:cfg.composition==="PARALLEL"?"psi + sum_s (U_s(psi)-psi)":cfg.composition==="SERIES"?
      "Ascending ordered full-gain composition":"Ascending half-gain composition followed by descending half-gain composition",
    input:cfg.state,active:cfg.active,time:cfg.time,source_time_step:K.SIM_DT,
    afterS,afterD,afterB:mstvq.psi,delta:mstvq.psi.map((v,i)=>K.Q.sub(v,cfg.state[i])),
    lambdaIn:K.inverseL(cfg.state),lambdaAfterCore:K.inverseL(afterS),lambdaOut:K.inverseL(mstvq.psi),
    kappaFeed:K.Q.mul(mstvq.trace,K.TGCR_KAPPA_FEED),mstvq,stepError:run.ledger,
    common_rotation_matchings:K.BOOK.MATCHINGS,sutra_records:records,distributed_grvq_records:grvqRecords,
    method_consumption:cfg.active.map(s=>({id:s.id,type:K.SUTRA_TYPE[s.id],
      core_sequences:records.filter(r=>r.id===s.id).map(r=>r.sequence),
      book_term_counts:records.filter(r=>r.id===s.id).map(r=>r.terms.length),
      direct_grvq:K.GRVQ_K[s.id]!==undefined,distributed_grvq_sequences:grvqRecords.filter(r=>r.id===s.id).map(r=>r.sequence),
      final_consumer:"afterB field -> inverseL -> quantum Hamiltonian coefficients"})),
    worker_receipts:workerReceipts,energy_before_exact:K.energy(cfg.state),energy_after_exact:K.energy(mstvq.psi),
    final_complexity:complexity(K,mstvq.psi),elapsed_ms:performance.now()-started};
  journal({stage:"source_formula_complete",afterB:result.afterB,lambdaOut:result.lambdaOut,mstvq,elapsed_ms:result.elapsed_ms});
  progress({stage:"complete",elapsed_ms:result.elapsed_ms,complexity:result.final_complexity});
  return options.returnRaw?result:encode(K,result,true);
}

async function executeBatch(requests,options={}){
  if(!Array.isArray(requests)||!requests.length)throw new TypeError("Supply independent requests");
  const start=performance.now(),backend=options.backend||"serial",workers=options.workers||2;
  if(requests.some(r=>(r.arithmetic||"expanded_rational")!==(requests[0].arithmetic||"expanded_rational")))throw new RangeError("Batch arithmetic must be explicit and consistent");
  let results;
  if(backend==="serial"){
    results=[];for(const request of requests)results.push(await execute(request,{onProgress:options.onProgress}));
  }else if(backend==="workers"){
    if(requests.some(r=>r.backend&&r.backend!=="serial"))throw new RangeError("Independent-case concurrency requires serial case backends; nested pools are not substituted");
    if(!Number.isInteger(workers)||workers<1)throw new RangeError("Positive worker count required");
    const pool=new WorkerPool(workers,loadKernel({arithmetic:requests[0].arithmetic,dagOptions:requests[0].dag_options}));
    try{results=await Promise.all(requests.map(request=>pool.submit({kind:"case",request})));}finally{await pool.close();}
  }else throw new RangeError("Unknown batch scheduling backend");
  return {schema:"quanqonscious.latest-hybrid-batch.v2",source:SOURCE_INFO,independent_case_backend:backend,
    workers:backend==="workers"?workers:0,results,elapsed_ms:performance.now()-start};
}

if(!isMainThread&&workerData?.latestHybridWorker){
  const K=loadKernel({arithmetic:workerData.arithmetic,dagOptions:workerData.dagOptions,matrixRepresentation:workerData.matrixRepresentation});
  parentPort.on("message",async message=>{
    try{
      if(message.encoded){const ids=K.exactDag.importGraph(message.expression_graph);message=K.exactDag.decodeImported(message.encoded,ids);}
      if(message.kind==="branch"){
        const value=branch(K,message.state,message.sutra,message.time);
        parentPort.postMessage(K.exactDag?{value:K.exactDag.encode(value),expression_graph:K.exactDag.exportGraph()}:{value});
      }else if(message.kind==="case")parentPort.postMessage({value:await execute(message.request)});
      else throw new RangeError("Unknown worker task");
    }catch(error){parentPort.postMessage({error:{name:error.name,message:error.message,stack:error.stack}});}
  });
  parentPort.postMessage({ready:true});
}

async function main(argv){
  const option=name=>{const i=argv.indexOf(name);return i<0?undefined:argv[i+1];};
  const request=JSON.parse(fs.readFileSync(option("--input")||0,"utf8")),output=option("--output");
  const onProgress=argv.includes("--progress")?value=>process.stderr.write(JSON.stringify(value)+"\n"):undefined;
  const result=request.requests?await executeBatch(request.requests,{backend:request.batch_backend,workers:request.workers,onProgress}):await execute(request,{onProgress});
  const text=JSON.stringify(result,null,2)+"\n";
  if(output){fs.mkdirSync(path.dirname(path.resolve(output)),{recursive:true});const data=output.endsWith(".gz")?zlib.gzipSync(text,{mtime:0}):text;fs.writeFileSync(output+".part",data);fs.renameSync(output+".part",output);process.stdout.write(JSON.stringify({output,bytes:Buffer.byteLength(data),sha256:hash(data),elapsed_ms:result.elapsed_ms})+"\n");}
  else process.stdout.write(text);
}
module.exports={loadKernel,execute,executeBatch,SOURCE_INFO,COMPOSITIONS,encode,complexity};
if(isMainThread&&require.main===module)main(process.argv.slice(2)).catch(error=>{process.stderr.write(JSON.stringify({status:"failed",error:error.name,message:error.message,stack:error.stack})+"\n");process.exitCode=1;});
