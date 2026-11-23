"use strict";
const fs=require("node:fs"),path=require("node:path"),zlib=require("node:zlib"),crypto=require("node:crypto"),assert=require("node:assert/strict");
const E=require("./latest_hybrid_executor.cjs");
const hash=x=>crypto.createHash("sha256").update(x).digest("hex");

async function verify(output=path.join(__dirname,"latest_hybrid_validation")){
  fs.mkdirSync(output,{recursive:true});const files=[];
  const save=(name,value)=>{const data=zlib.gzipSync(JSON.stringify(value)+"\n",{mtime:0});fs.writeFileSync(path.join(output,name),data);files.push({path:name,bytes:data.length,sha256:hash(data)});};
  const expanded=E.loadKernel({matrixRepresentation:"source_dense"}),dag=E.loadKernel({arithmetic:"exact_dag"});
  const a=expanded.initState(),b=dag.initState(),rows=[];
  const longPeriods=[];
  for(const [id,denominator] of [[1,4269n],[12,4267n]]){
    const terms=expanded.BOOK.terms(id,{denominator:expanded.Q.mk(denominator)});
    const digits=[];let remainder=1n;
    do{const numerator=10n*remainder;digits.push(numerator/denominator);remainder=numerator%denominator;}while(remainder!==1n);
    assert.equal(terms.length,digits.length);
    for(let i=0;i<digits.length;i++){assert.equal(terms[i].n,digits[i]);assert.equal(terms[i].d,1n);}
    const numerator=BigInt(digits.map(String).join(""));
    assert.equal(numerator*denominator,10n**BigInt(digits.length)-1n);
    const field=a.slice();field[0]=expanded.Q.mk(-426n);
    assert.equal(expanded.BOOK.operandsOf(id,field).denominator.n,denominator);
    longPeriods.push({id,denominator:denominator.toString(),complete_period_length:terms.length,
      digits:digits.map(String).join(""),independent_long_division_equal:true,exact_reconstruction_equal:true});
  }
  for(let id=1;id<=29;id++){
    const raw=[],compressed=[];
    const expected=expanded.applySutra(id,a,expanded.Q.parse("1/29"),{t:expanded.Q.ZERO,record:raw});
    const actual=dag.applySutra(id,b,dag.Q.parse("1/29"),{t:dag.Q.ZERO,record:compressed});
    for(let vertex=0;vertex<16;vertex++)assert(dag.Q.eq(actual[vertex],dag.Q.mk(expected[vertex].n,expected[vertex].d)),`S${id} vertex ${vertex}`);
    assert.equal(JSON.stringify(expanded.serializeExact(raw[0].info.operands)),JSON.stringify(expanded.serializeExact(compressed[0].info.operands)));
    assert.equal(JSON.stringify(expanded.serializeExact(raw[0].info.terms)),JSON.stringify(expanded.serializeExact(compressed[0].info.terms)));
    rows.push({id,exact_equal_vertices:16,operands:expanded.serializeExact(raw[0].info.operands),terms:expanded.serializeExact(raw[0].info.terms),expected_output:expanded.serializeExact(expected),actual_output:dag.exactDag.encode(actual)});
  }
  save("individual_source_parity.json.gz",{source:E.SOURCE_INFO,arithmetic_sha256:dag.arithmetic_sha256,comparisons:rows,expression_graph:dag.exactDag.exportGraph()});
  const modeResults={};
  for(const composition of E.COMPOSITIONS){
    const result=await E.execute({initial_state:"source",strength:"1/29",composition,arithmetic:"exact_dag",time:"0"});
    const core=composition==="SYMMETRIC_CONCURRENT"?58:29;
    assert.equal(result.sutra_records.length,core);assert.equal(result.distributed_grvq_records.length,25);
    assert.deepEqual(result.method_consumption.map(x=>x.id),Array.from({length:29},(_,i)=>i+1));
    assert.equal(result.intervals_used,false);modeResults[composition]=result;
    save(composition+".json.gz",result);
  }
  const parallel=await E.execute({initial_state:"source",strength:"1/29",composition:"PARALLEL",arithmetic:"exact_dag",backend:"workers",workers:2,time:"0"});
  assert.equal(JSON.stringify(parallel.final_complexity.state_root_digests),JSON.stringify(modeResults.PARALLEL.final_complexity.state_root_digests));
  const workerIds=[...new Set(parallel.worker_receipts.map(r=>r.worker_thread_id))];assert.equal(workerIds.length,2);
  save("PARALLEL_workers.json.gz",parallel);
  const modeDigests=E.COMPOSITIONS.map(mode=>modeResults[mode].final_complexity.state_root_digests.join(":"));
  assert.equal(new Set(modeDigests).size,3);
  const batchRequests=E.COMPOSITIONS.map(composition=>({initial_state:"source",strength:"1/58",composition,arithmetic:"exact_dag",backend:"serial",time:"0"}));
  const batch=await E.executeBatch(batchRequests,{backend:"workers",workers:2});
  assert.equal(batch.results.length,3);
  for(let i=0;i<3;i++)assert.equal(batch.results[i].mathematical_composition,E.COMPOSITIONS[i]);
  save("independent_modes_concurrent.json.gz",batch);
  const result={schema:"grvq-latest-source-validation/v2",status:"complete",source:E.SOURCE_INFO,
    arithmetic_sha256:dag.arithmetic_sha256,executor_sha256:hash(fs.readFileSync(path.join(__dirname,"latest_hybrid_executor.cjs"))),
    exact_source_value_equalities:29*16,identical_operand_sets:29,identical_term_sets:29,
    domain_extensions:dag.domain_extensions,long_period_checks:longPeriods,
    complete_modes:Object.fromEntries(E.COMPOSITIONS.map(mode=>[mode,{core_applications:modeResults[mode].sutra_records.length,
      distributed_grvq_applications:modeResults[mode].distributed_grvq_records.length,elapsed_ms:modeResults[mode].elapsed_ms,
      expression_nodes:modeResults[mode].final_complexity.nodes,interval_values_in_state:0}])),
    parallel_serial_worker_exact_digest_equality:true,actual_worker_ids:workerIds,
    three_mathematical_outputs_distinct:true,independent_mode_concurrency:{backend:"workers",workers:2,completed_modes:3,elapsed_ms:batch.elapsed_ms},files};
  fs.writeFileSync(path.join(output,"validation.json"),JSON.stringify(result,null,2)+"\n");
  return result;
}
module.exports={verify};
if(require.main===module)verify(process.argv[2]).then(result=>process.stdout.write(JSON.stringify(result,null,2)+"\n")).catch(error=>{console.error(error);process.exitCode=1;});
