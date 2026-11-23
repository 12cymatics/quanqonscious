"use strict";
const fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict"),crypto=require("node:crypto");
const E=require("./latest_hybrid_executor.cjs");
const hash=x=>crypto.createHash("sha256").update(x).digest("hex");

function verify(output=path.join(__dirname,"latest_hybrid_validation","matrix_representation.json")){
  const dense=E.loadKernel({matrixRepresentation:"source_dense"}),tensor=E.loadKernel({matrixRepresentation:"commuting_tensor"});
  const cases=[],strengths=["1/29","-2/7"],state=dense.initState();
  let equalEntries=0;
  function compare(label,terms,strength){
    const a=dense.BOOK.matrixOf(terms,dense.Q.parse(strength)),b=tensor.BOOK.matrixOf(terms,tensor.Q.parse(strength));
    assert.equal(a.D,b.D,`${label}: unchanged unreduced denominator`);
    for(let i=0;i<16;i++)for(let j=0;j<16;j++){
      assert.equal(a.N[i][j],b.N[i][j],`${label} N[${i}][${j}]`);
      assert.equal(a.N[i][j]*b.D,b.N[i][j]*a.D,`${label} exact rational equality`);equalEntries++;
    }
    assert.equal(b.tensor_certificate.visited_terms,terms.length);
    assert.equal(b.tensor_certificate.grouped_multiplicities.reduce((a,b)=>a+b,0),b.tensor_certificate.nonzero_terms);
    cases.push({label,strength,terms:dense.serializeExact(terms),equal_integer_entries:256,identical_unreduced_denominator:true});
  }
  for(let id=1;id<=29;id++)for(const strength of strengths)
    compare(`field-derived S${id}`,dense.BOOK.terms(id,dense.BOOK.operandsOf(id,state)),strength);
  for(const [label,values] of [
    ["signed distinct rational coefficients",["2/3","-7/5","0","13/11","2/3","-7/5"]],
    ["all source identity coefficients",["0","0","0"]],
    ["quarter-turn signs and orientation",["1","-1","2","-3","5"]],
    ["empty exact identity",[]]])
    for(const strength of ["1","-3/4","0"])compare(label,values.map(x=>dense.Q.parse(x)),strength);
  const denominator=3709n,terms=tensor.BOOK.terms(1,{denominator:tensor.Q.mk(denominator)});
  const independent=[];let remainder=1n;
  do{const next=10n*remainder;independent.push(next/denominator);remainder=next%denominator;}while(remainder!==1n);
  assert.equal(terms.length,3708);assert.equal(terms.length,independent.length);
  for(let i=0;i<terms.length;i++){assert.equal(terms[i].n,independent[i]);assert.equal(terms[i].d,1n);}
  assert.equal(BigInt(independent.map(String).join(""))*denominator,10n**BigInt(terms.length)-1n);
  const strength=tensor.Q.parse("1/58"),start=performance.now(),whole=tensor.BOOK.matrixOf(terms,strength),elapsed=performance.now()-start;
  const split=1733,left=tensor.BOOK.matrixOf(terms.slice(0,split),strength).tensor_certificate,
    right=tensor.BOOK.matrixOf(terms.slice(split),strength).tensor_certificate,all=whole.tensor_certificate;
  const [a,b]=left.single_axis_numerator,[c,d]=right.single_axis_numerator,[C,S]=all.single_axis_numerator;
  assert.equal(C,a*c-b*d);assert.equal(S,a*d+b*c);
  assert.equal(all.single_axis_denominator,left.single_axis_denominator*right.single_axis_denominator);
  assert.equal(C*C+S*S,all.single_axis_denominator**2n);
  assert.equal(whole.D,all.single_axis_denominator**4n);
  assert.equal(all.visited_terms,3708);assert.equal(all.nonzero_terms+all.zero_identity_terms,3708);
  assert.equal(all.grouped_multiplicities.reduce((a,b)=>a+b,0),all.nonzero_terms);
  const proof={schema:"exact-book-matrix-representation-proof/v1",status:"complete",source:E.SOURCE_INFO,
    executor_sha256:hash(fs.readFileSync(path.join(__dirname,"latest_hybrid_executor.cjs"))),
    identity:"For every exact t=p/q, normalize (c,s,d)=(q^2-p^2,2pq,q^2+p^2) by the same source gcd. Each of the four complete oriented hypercube axes occurs twice. Their rotations commute, so C+iS=product_terms(c+i*s)^2 and D1=product_terms d^2. The original N[i,j]=(-1)^popcount((~i)&j&15)*C^(4-popcount(i xor j))*S^popcount(i xor j), and original D=D1^4. Grouping identical factors retains every coefficient multiplicity.",
    matching_axes:all.matching_axes,axis_multiplicities:all.axis_multiplicities,
    representation_selection:"Universal deterministic tensor representation; source_dense is an explicit validation oracle, never a workload fallback",
    exact_integer_matrix_entry_equalities:equalEntries,exact_cross_multiplied_matrix_entry_equalities:equalEntries,bounded_cases:cases,
    long_period:{sutra:1,denominator:denominator.toString(),coefficient_count:terms.length,
      independent_long_division_equal:true,exact_repetend_reconstruction:true,
      digits_sha256:hash(independent.map(String).join("")),matrix_construction_ms:elapsed,
      denominator_bits:whole.D.toString(2).length,visited_terms:all.visited_terms,
      zero_identity_terms:all.zero_identity_terms,nonzero_terms:all.nonzero_terms,
      source_matching_rotations:all.total_source_matching_rotations,grouped_multiplicities:all.grouped_multiplicities,
      exact_gaussian_norm_identity:true,exact_split_recomposition_identity:true,split_after_coefficient:split,
      no_coefficients_omitted:true,unreduced_source_denominator_preserved:true}};
  fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(proof,null,2)+"\n");
  return {output,status:proof.status,equalities:equalEntries,long_period:proof.long_period};
}
module.exports={verify};
if(require.main===module)console.log(JSON.stringify(verify(process.argv[2]),null,2));
