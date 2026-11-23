"use strict";

// Exact quantum moments remain small rational numbers. The evolving GRVQ
// coefficients stay in the source kernel's exact expression arena throughout.
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const zlib = require("node:zlib");
const { performance } = require("node:perf_hooks");
const E = require("./latest_hybrid_executor.cjs");
const hash = x => crypto.createHash("sha256").update(x).digest("hex");
const QUANTUM_SOURCE_SHA256 = "1ea3d56f01541181ba2ffafa536f00405198afcfddc5518e3a2e29f8731aeccd";
const abs = x => x < 0n ? -x : x;
function gcd(a,b) { a=abs(a);b=abs(b);while(b)[a,b]=[b,a%b];return a; }
function mk(n,d=1n) {
  if(typeof n!=="bigint"||typeof d!=="bigint")throw new TypeError("Rational operands must use BigInt");
  if(d===0n)throw new RangeError("Exact division by zero");if(d<0n){n=-n;d=-d;}
  const g=gcd(n,d);return {n:n/g,d:d/g};
}
const ZERO=mk(0n),ONE=mk(1n),TWO=mk(2n),HALF=mk(1n,2n);
function parse(x) { if(typeof x!=="string"||!/^[+-]?\d+(?:\/[+-]?\d+)?$/.test(x))throw new TypeError("Use an exact rational string");const a=x.split("/");return mk(BigInt(a[0]),a[1]===undefined?1n:BigInt(a[1])); }
const textQ = q => q.d===1n?q.n.toString():`${q.n}/${q.d}`;
function add(a,b) { const g=gcd(a.d,b.d);return mk(a.n*(b.d/g)+b.n*(a.d/g),(a.d/g)*b.d); }
const neg = a => ({n:-a.n,d:a.d});
const sub = (a,b) => add(a,neg(b));
function mul(a,b) { const g=gcd(a.n,b.d),h=gcd(b.n,a.d);return {n:(a.n/g)*(b.n/h),d:(a.d/h)*(b.d/g)}; }
const div = (a,b) => { if(!b.n)throw new RangeError("Exact division by zero");return mul(a,mk(b.d,b.n)); };
const eq = (a,b) => a.n===b.n&&a.d===b.d;
const sum = a => a.reduce(add,ZERO);
const scalarQ = (Q,a) => Q.mk(a.n,a.d);

function circuitState(parameters) {
  if(!Array.isArray(parameters)||parameters.length!==12)throw new RangeError("The source circuit has exactly twelve Ry parameters");
  parameters=parameters.map(p=>typeof p==="string"?parse(p):mk(p.n,p.d));
  let state=Array.from({length:16},(_,i)=>i===0?ONE:ZERO);
  for(let layer=0;layer<3;layer++) {
    for(let q=0;q<4;q++) {
      const t=parameters[layer*4+q],tt=mul(t,t),den=add(ONE,tt),c=div(sub(ONE,tt),den),s=div(mul(TWO,t),den);
      const next=state.slice(),bit=1<<q;
      for(let i=0;i<16;i++)if(!(i&bit)) { const j=i|bit,a=state[i],b=state[j];next[i]=sub(mul(c,a),mul(s,b));next[j]=add(mul(s,a),mul(c,b)); }
      state=next;
    }
    for(let control=0;control<3;control++) {
      const next=new Array(16);
      for(let i=0;i<16;i++)next[(i&(1<<control))?i^(1<<(control+1)):i]=state[i];
      state=next;
    }
  }
  return state;
}

function moments(state) {
  if(!Array.isArray(state)||state.length!==16)throw new RangeError("The circuit state must have sixteen amplitudes");
  const probabilities=state.map(a=>mul(a,a)),norm=sum(probabilities);
  if(!eq(norm,ONE))throw new Error("The exact Ry/CNOT circuit norm is not one");
  const expectation=mask=>sum(probabilities.map((p,i)=>((i&mask).toString(2).split("1").length-1)%2?neg(p):p));
  const z=[0,1,2,3].map(q=>expectation(1<<q)),zz=[0,1,2].map(q=>expectation((1<<q)|(1<<(q+1))));
  return {state,probabilities,norm,z,zz,constant:mul(HALF,sum(zz))};
}

function bitCoefficients(index) {
  const z=[0,1,2,3].map(q=>(index&(1<<q))?neg(ONE):ONE);
  const zz=[0,1,2].map(q=>mul(z[q],z[q+1]));
  return {z,zz,constant:mul(HALF,sum(zz))};
}
function energy(Q,lambda,m) {
  if(lambda.length!==4)throw new RangeError("The Hamiltonian requires all four source lambda coefficients");
  return m.z.reduce((e,z,q)=>Q.add(e,Q.mul(lambda[q],scalarQ(Q,z))),scalarQ(Q,m.constant));
}
function difference(Q,lambda,a,b) {
  const z=a.z.map((x,q)=>sub(x,b.z[q])),constant=sub(a.constant,b.constant);
  const value=z.reduce((e,c,q)=>Q.add(e,Q.mul(lambda[q],scalarQ(Q,c))),scalarQ(Q,constant));
  return {value,z,constant,identically_zero:z.every(q=>q.n===0n)&&constant.n===0n};
}
function compareEnergy(Q,lambda,a,b) {
  const d=difference(Q,lambda,a,b);
  return {sign:d.identically_zero?0:Q.sign(d.value),difference:d.value,
    proof:d.identically_zero?"identical exact rational Pauli coefficients":"certified exact expression comparison"};
}
function projected(arena,q) {
  const b=arena.enclosure(q,128),den=2**128;
  const lo=Number(b.lo)/den,hi=Number(b.hi)/den;
  if(!Number.isFinite(lo)||!Number.isFinite(hi))throw new RangeError("The diagnostic decimal projection is outside Number range");
  return {value:(lo+hi)/2,certificate:{lower_integer:b.lo.toString(),upper_integer:b.hi.toString(),denominator_power:128},purpose:"display only; never an arithmetic input"};
}

function independentDiagnostic(K,lambda,m) {
  const Q=K.Q,levels=Array.from({length:16},(_,i)=>bitCoefficients(i));
  let ground=0;
  for(let i=1;i<16;i++)if(compareEnergy(Q,lambda,levels[i],levels[ground]).sign<0)ground=i;
  const groundIndices=levels.flatMap((b,i)=>compareEnergy(Q,lambda,b,levels[ground]).sign===0?[i]:[]);
  // Collect the full 16-outcome Rayleigh expression by its independent lambda
  // coefficients before comparison. This proves equality for every lambda.
  const z=[0,1,2,3].map(q=>sum(levels.map((b,i)=>mul(m.probabilities[i],b.z[q]))));
  const c=sum(levels.map((b,i)=>mul(m.probabilities[i],b.constant)));
  if(!z.every((x,q)=>eq(x,m.z[q]))||!eq(c,m.constant))throw new Error("The full sixteen-outcome and Pauli contractions disagree");
  const gap=difference(Q,lambda,m,levels[ground]);
  const gapSign=gap.identically_zero?0:Q.sign(gap.value);
  if(gapSign<0)throw new Error("The variational state is below the exact diagonal ground energy");
  const fidelity=sum(groundIndices.map(i=>m.probabilities[i]));
  const eigenvalues=levels.map(b=>energy(Q,lambda,b));
  return {method:"exact minimum of all sixteen diagonal energies; diagnostic only",
    ground_indices:groundIndices,ground_energy:eigenvalues[ground],eigenvalues,
    energy_gap:gap.value,energy_gap_is_exactly_zero:gapSign===0,ground_fidelity_rational:textQ(fidelity),
    full_sixteen_outcome_rayleigh_identity:{proved:true,coefficient_field:"Q",lambda_coefficients:z.map(textQ),constant:textQ(c)},
    display:{ground_energy:projected(K.exactDag,eigenvalues[ground]),energy_gap:projected(K.exactDag,gap.value)}};
}

function coordinateSearch(K,lambda,initial,{sweeps,gridDenominator,onProgress=()=>{}}) {
  const Q=K.Q,parameters=initial.map(parse),grid=Array.from({length:2*gridDenominator+1},(_,i)=>mk(BigInt(i-gridDenominator),BigInt(gridDenominator)));
  let current=moments(circuitState(parameters));const first=current,initialEnergy=energy(Q,lambda,current),lines=[];
  const cache=new Map(),costs={circuit_evaluations:1,candidate_energy_comparisons:0,coordinate_lines:0,accepted_updates:0};
  cache.set(parameters.map(textQ).join(","),current);
  for(let sweep=0;sweep<sweeps;sweep++)for(let coordinate=0;coordinate<12;coordinate++) {
    const old=parameters[coordinate],oldMoments=current,oldEnergy=energy(Q,lambda,current),trials=[];
    let winner=old,best=current;
    const candidates=[old,...grid.filter(t=>!eq(t,old))];
    for(const t of candidates) {
      const proposed=parameters.slice();proposed[coordinate]=t;
      const key=proposed.map(textQ).join(",");let candidate=cache.get(key);
      if(!candidate){candidate=moments(circuitState(proposed));cache.set(key,candidate);costs.circuit_evaluations++;}
      const comparison=compareEnergy(Q,lambda,candidate,best);costs.candidate_energy_comparisons++;
      const candidateEnergy=energy(Q,lambda,candidate);
      trials.push({parameter:textQ(t),energy:candidateEnergy,z:candidate.z.map(textQ),zz:candidate.zz.map(textQ),
        comparison_with_incumbent:comparison.sign,comparison_difference:comparison.difference,comparison_proof:comparison.proof});
      if(comparison.sign<0){winner=t;best=candidate;}
    }
    parameters[coordinate]=winner;current=best;
    const change=compareEnergy(Q,lambda,current,oldMoments);
    if(change.sign>0)throw new Error("An exact coordinate update increased the frozen-Hamiltonian energy");
    if(!eq(old,winner))costs.accepted_updates++;
    const line={sweep,coordinate,parameter_before:textQ(old),parameter_after:textQ(winner),energy_before:oldEnergy,
      energy_after:energy(Q,lambda,current),energy_change:change.difference,trials};
    lines.push(line);costs.coordinate_lines++;
    onProgress({stage:"quantum_coordinate",sweep,coordinate,costs:{...costs}});
  }
  return {parameters:parameters.map(textQ),moments:current,initial_moments:first,initial_energy:initialEnergy,
    energy:energy(Q,lambda,current),lines,costs,completed_sweeps:sweeps,
    stop_reason:"completed every declared sweep and coordinate",
    search_space:{kind:"explicit finite exact rational coordinate grid",grid:grid.map(textQ),include_current:true,
      ties:"retain incumbent",global_continuous_optimum_claim:false}};
}

function validateConfiguration(input) {
  if(!input||typeof input!=="object"||Array.isArray(input))throw new TypeError("Supply a trajectory configuration");
  const c={...input};
  if(!E.COMPOSITIONS.includes(c.composition))throw new RangeError("Unknown mathematical composition");
  for(const k of ["steps","sweeps","grid_denominator","branch_workers"])if(!Number.isInteger(c[k])||c[k]<1)throw new RangeError(`${k} must be a positive integer`);
  if(!Number.isInteger(c.seed))throw new TypeError("Seed must be an integer");
  if(typeof c.feedback!=="boolean")throw new TypeError("Feedback selection must be explicit");
  if(!["serial","workers"].includes(c.backend))throw new RangeError("Select serial or workers backend");
  if(c.backend==="workers"&&c.composition!=="PARALLEL")throw new RangeError("Within-field workers require PARALLEL composition");
  const strength=parse(c.strength);if(strength.n<=0n||strength.n>strength.d)throw new RangeError("All29 require nonzero strength in (0,1]");
  if(!Array.isArray(c.initial_parameters)||c.initial_parameters.length!==12)throw new RangeError("Supply twelve exact initial Cayley parameters");
  c.initial_parameters=c.initial_parameters.map(x=>textQ(parse(x)));
  if(!c.case_id||!/^[-A-Za-z0-9_]+$/.test(c.case_id))throw new RangeError("Use a safe nonempty case id");
  return c;
}
function atomicJSON(filename,value,gzip=false) {
  const text=JSON.stringify(value,null,2)+"\n",data=gzip?zlib.gzipSync(text,{level:6,mtime:0}):text;
  fs.writeFileSync(filename+".tmp",data);fs.renameSync(filename+".tmp",filename);
}
function quantumRecord(m) { return {amplitudes:m.state.map(textQ),probabilities:m.probabilities.map(textQ),z:m.z.map(textQ),zz:m.zz.map(textQ),norm:textQ(m.norm)}; }


/* The mathematical prefix above is recovered verbatim (SHA256
 * 6206bef66d8105a4e5081cd9e5188fbb101b7f51da273cfc2bfaef21d65765ea).
 * The remaining trajectory I/O was rebuilt after the original host became
 * unavailable. This is a new source revision; all current results are executed.
 */
const PREVIOUS_IMPLEMENTATION_SHA256="3f9980523a0724615ea0ffd8cee26ac36afaf3f535fcad0b5bc1415ff0f30c01";
const text=textQ,adopt=scalarQ;
const rational={mk,parse,add,sub,mul,div,neg,eq,text:textQ,ZERO,ONE};
class ArithmeticError extends Error{constructor(message){super(message);this.name="ArithmeticError";}}

function verifyProjectionBasis(K){
  const zero=Array.from({length:16},()=>K.Q.ZERO),basis=[];
  for(let q=0;q<4;q++)basis.push(K.injectLambda(zero,q,K.Q.ONE));
  const projections=basis.map(v=>K.inverseL(v));
  for(let q=0;q<4;q++)for(let k=0;k<4;k++)
    if(!K.Q.eq(projections[q][k],q===k?K.Q.ONE:K.Q.ZERO))throw new ArithmeticError("Source feedback injection and inverseL are not dual");
  return {kind:"exact source injection/projection dual basis identity",basis,projections};
}
async function runHybridCase(configuration,{outputDirectory,onProgress=()=>{}}){
  const cfg=validateConfiguration(configuration),directory=path.resolve(outputDirectory);
  if(fs.existsSync(directory)&&fs.readdirSync(directory).length)
    throw new Error("The existing output is preserved; use a new output directory.");
  fs.mkdirSync(directory,{recursive:true});atomicJSON(path.join(directory,"request.json"),cfg);
  const started=performance.now(),K=E.loadKernel({arithmetic:"exact_dag",dagOptions:{precisionBits:128,maxPrecisionBits:65536,leafBits:4096}});
  const arena=K.exactDag,Q=K.Q,journalPath=path.join(directory,"journal.jsonl");
  let fd=fs.openSync(journalPath,"wx"),closed=false;
  const writeJournal=value=>{fs.writeSync(fd,JSON.stringify(arena.encode(value))+"\n");fs.fsyncSync(fd);};
  const code=Object.fromEntries(["hybrid_quantum.cjs","latest_hybrid_executor.cjs","exact_dag.cjs"].map(name=>[name,hash(fs.readFileSync(path.join(__dirname,name)))]));
  const report={schema:"grvq-quantum-feedback-case/v2",status:"running",configuration:cfg,source:E.SOURCE_INFO,
    arithmetic_module_sha256:K.arithmetic_sha256,quantum_source:{name:"vedic_v21 4.html",sha256:QUANTUM_SOURCE_SHA256,
      qubits:4,layers:3,cayley_parameters:12,initial_state:"0000",
      gate_order:"Ry q0,q1,q2,q3; CX0->1,CX1->2,CX2->3 after every layer including the last",
      hamiltonian:"sum(q=0..3,lambda[q]*Z[q]) + 1/2*sum(q=0..2,Z[q]*Z[q+1])"},
    reconstruction:{kind:"byte-recovered mathematical prefix with reconstructed trajectory I/O",
      previous_implementation_sha256:PREVIOUS_IMPLEMENTATION_SHA256,previous_results_not_recreated_by_assertion:true,
      current_results:"computed by this execution"},
    integration:{field:"complete source formulaOf stages",feedback:"exact -<Zq>/1000; disabled paired control applies zero",
      schedule:"one complete source formula, frozen-H coordinate optimization, exact feedback, source SIM_DT advance",
      diagnostic:"sixteen diagonal basis energies are diagnostic only"},
    created_utc:new Date().toISOString(),runtime:{node:process.version,platform:process.platform,architecture:process.arch},
    code_sha256:code,steps:[],totals:{core_method_calls:0,distributed_grvq_updates:0,coordinate_lines:0,
      circuit_evaluations:0,candidate_energy_comparisons:0,accepted_updates:0}};
  let field=K.initState(),time=Q.ZERO,parameters=cfg.initial_parameters.slice();
  try{
    const projectionProof=verifyProjectionBasis(K);
    for(let step=0;step<cfg.steps;step++){
      onProgress({stage:"outer_step_start",step,case_id:cfg.case_id});
      const request={case_id:cfg.case_id,composition:cfg.composition,strength:cfg.strength,time,
        backend:cfg.backend,workers:cfg.branch_workers,arithmetic:"exact_dag",exact_bits:null,require_all_29:true};
      if(field===null)request.initial_state="source";else request.state=field;
      const formula=await E.execute(request,{kernel:K,rawState:true,returnRaw:true,
        onJournal:r=>writeJournal({step,source:r}),onProgress:p=>onProgress({...p,step,case_id:cfg.case_id})});
      const lambda=formula.lambdaOut;
      const inputDigests=Array.from(formula.input,x=>arena.digest(x));
      if(step&&JSON.stringify(inputDigests)!==JSON.stringify(report.steps[step-1].next_field_digests))
        throw new ArithmeticError("The source step did not consume the preceding quantum feedback field");
      const optimized=coordinateSearch(K,lambda,parameters,{sweeps:cfg.sweeps,gridDenominator:cfg.grid_denominator,
        onProgress:p=>onProgress({...p,step,case_id:cfg.case_id})});
      const diagnostic=independentDiagnostic(K,lambda,optimized.moments);
      const computed=optimized.moments.z.map(z=>div(neg(z),mk(1000n)));
      const applied=computed.map(x=>cfg.feedback?x:ZERO);
      let nextField=formula.afterB;
      for(let q=0;q<4;q++)nextField=K.injectLambda(nextField,q,adopt(Q,applied[q]));
      const actualNextLambda=K.inverseL(nextField),expectedNextLambda=lambda.map((x,q)=>Q.add(x,adopt(Q,applied[q])));
      const expectedCore=cfg.composition==="SYMMETRIC_CONCURRENT"?58:29;
      const all29=formula.method_consumption.length===29&&formula.method_consumption.every((x,i)=>x.id===i+1&&x.core_sequences.length===(expectedCore/29));
      if(!all29||formula.sutra_records.length!==expectedCore||formula.distributed_grvq_records.length!==25)
        throw new ArithmeticError("A required source contribution was not consumed");
      const quantum={parameters:optimized.parameters,amplitudes:optimized.moments.state.map(text),
        probabilities:optimized.moments.probabilities.map(text),z:optimized.moments.z.map(text),zz:optimized.moments.zz.map(text),
        norm:text(optimized.moments.norm),constant:text(optimized.moments.constant)};
      const quantumFile="quantum_"+String(step).padStart(3,"0")+".json";
      const frameFile="frame_"+String(step).padStart(3,"0")+".json.gz";
      const frame={schema:"grvq-quantum-feedback-frame/v2",step,time,source_formula:formula,
        quantum:{...quantum,coordinate_search:optimized.search_space,coordinate_records:optimized.lines,diagnostic},
        feedback:{enabled:cfg.feedback,law:"delta_lambda[q] = -<Zq>/1000",
          computed_delta_rationals:computed.map(text),applied_delta_rationals:applied.map(text),
          next_field:nextField,actual_next_lambda:actualNextLambda,expected_next_lambda:expectedNextLambda,
          projection_identity:projectionProof,proof:"linearity and the exact source dual basis identity; no expanded-field equality shortcut"}};
      atomicJSON(path.join(directory,quantumFile),quantum);
      atomicJSON(path.join(directory,frameFile),arena.encode(frame),true);
      const rawSummary={step,time,frame_file:frameFile,quantum_file:quantumFile,
        input_field_digests:inputDigests,after_formula_field_digests:Array.from(formula.afterB,x=>arena.digest(x)),
        next_field_digests:Array.from(nextField,x=>arena.digest(x)),
        lambda_in:formula.lambdaIn,lambda_out:lambda,lambda_after_feedback:actualNextLambda,
        energy_initial:optimized.initial_energy,energy_final:optimized.energy,
        exact_ground_energy:diagnostic.ground_energy,energy_gap:diagnostic.energy_gap,energy_gap_is_exactly_zero:diagnostic.energy_gap_is_exactly_zero,
        ground_fidelity_rational:diagnostic.ground_fidelity_rational,feedback_deltas:applied.map(text),
        computed_feedback_deltas:computed.map(text),core_method_calls:formula.sutra_records.length,
        distributed_grvq_updates:formula.distributed_grvq_records.length,all29_consumed:all29,quantum_costs:optimized.costs,
        display:{lambda_in:Array.from(formula.lambdaIn,x=>projected(arena,x)),
          lambda_out:Array.from(lambda,x=>projected(arena,x)),lambda_after_feedback:Array.from(actualNextLambda,x=>projected(arena,x)),
          energy_initial:projected(arena,optimized.initial_energy),energy_final:projected(arena,optimized.energy),
          ground_energy:projected(arena,diagnostic.ground_energy),energy_gap:projected(arena,diagnostic.energy_gap)}};
      const summary=arena.encode(rawSummary);report.steps.push(summary);
      report.totals.core_method_calls+=summary.core_method_calls;
      report.totals.distributed_grvq_updates+=summary.distributed_grvq_updates;
      for(const [key,value]of Object.entries(optimized.costs))report.totals[key]+=value;
      field=nextField;parameters=optimized.parameters;time=Q.add(time,K.SIM_DT);
      atomicJSON(path.join(directory,"checkpoint.json"),arena.encode({schema:"grvq-quantum-feedback-checkpoint/v2",
        next_step:step+1,configuration:cfg,field,time,parameters:parameters}));
      report.elapsed_seconds=(performance.now()-started)/1000;report.exact_arithmetic_statistics=arena.stats();
      atomicJSON(path.join(directory,"exact_graph.json.gz"),arena.exportGraph(),true);
      atomicJSON(path.join(directory,"result.json"),report);
      // Summary is already encoded: do not attempt to adopt cloned references.
      fs.writeSync(fd,JSON.stringify({stage:"outer_step_complete",step,summary})+"\n");fs.fsyncSync(fd);
      onProgress({stage:"outer_step_complete",step,case_id:cfg.case_id,elapsed_seconds:report.elapsed_seconds,display:summary.display});
    }
    fs.closeSync(fd);closed=true;
    fs.writeFileSync(journalPath+".gz",zlib.gzipSync(fs.readFileSync(journalPath),{level:6,mtime:0}));fs.unlinkSync(journalPath);
    report.status="complete";report.elapsed_seconds=(performance.now()-started)/1000;
    report.exact_arithmetic_statistics=arena.stats();
    report.artifact_sha256=Object.fromEntries(fs.readdirSync(directory).filter(name=>name!=="result.json"&&!name.endsWith(".tmp"))
      .map(name=>[name,hash(fs.readFileSync(path.join(directory,name)))]));
    atomicJSON(path.join(directory,"result.json"),report);return report;
  }catch(error){
    if(!closed){fs.closeSync(fd);closed=true;}
    report.status="failed";report.error={name:error.name,message:error.message,stack:error.stack};
    report.elapsed_seconds=(performance.now()-started)/1000;report.exact_arithmetic_statistics=arena.stats();
    atomicJSON(path.join(directory,"exact_graph.json.gz"),arena.exportGraph(),true);
    atomicJSON(path.join(directory,"result.json"),report);throw error;
  }
}
async function main(argv){
  const option=name=>{const i=argv.indexOf(name);return i<0?undefined:argv[i+1];};
  const input=option("--input"),output=option("--output");if(!input||!output)throw new Error("Use --input request.json --output a-new-case-directory");
  const result=await runHybridCase(JSON.parse(fs.readFileSync(input,"utf8")),{outputDirectory:output,
    onProgress:argv.includes("--progress")?p=>process.stdout.write(JSON.stringify(p)+"\n"):undefined});
  process.stdout.write(JSON.stringify({status:result.status,steps:result.steps.length,elapsed_seconds:result.elapsed_seconds,
    result:path.resolve(output,"result.json")})+"\n");
}
module.exports={circuitState,moments,energy,difference,compareEnergy,independentDiagnostic,coordinateSearch,
  projected,runHybridCase,validateConfiguration,rational,QUANTUM_SOURCE_SHA256};
if(require.main===module)main(process.argv.slice(2)).catch(error=>{
  process.stderr.write(JSON.stringify({status:"failed",error:error.name,message:error.message,stack:error.stack})+"\n");process.exitCode=1;
});
