"""Independent Fraction references for exact arithmetic and certificates."""
from fractions import Fraction
import json
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_graph_arithmetic_against_fraction():
    rng = random.Random(1981)
    values = [Fraction(rng.randrange(-100000, 100000), rng.randrange(1, 100000))
              for _ in range(20)]
    rows = [{"op": "q", "n": str(x.numerator), "d": str(x.denominator)} for x in values]
    for index in range(220):
        available = [i for i, x in enumerate(values)
                     if x.numerator.bit_length() + x.denominator.bit_length() < 1200]
        a, b = rng.choice(available), rng.choice(available)
        op = ("add", "sub", "mul", "div")[index % 4]
        if op == "div" and values[b] == 0:
            b = 0
        if op == "add":
            result = values[a] + values[b]
        elif op == "sub":
            result = values[a] - values[b]
        elif op == "mul":
            result = values[a] * values[b]
        else:
            result = values[a] / values[b]
        rows.append({"op": op, "a": a, "b": b,
                     "n": str(result.numerator), "d": str(result.denominator)})
        values.append(result)
    script = r'''
const fs=require('node:fs'),assert=require('node:assert/strict');
const {createExactDag}=require('./experiments/grvq_ansatz_comparison/exact_dag.cjs');
const graph=createExactDag({leafBits:32}),Q=graph.Q,values=[];
for(const row of JSON.parse(fs.readFileSync(0,'utf8'))){
 const value=row.op==='q'?Q.mk(BigInt(row.n),BigInt(row.d)):Q[row.op](values[row.a],values[row.b]);
 const n=BigInt(row.n),d=BigInt(row.d),bounds=graph.enclosure(value,256),scale=1n<<256n;
 assert(bounds.lo*d<=n*scale&&bounds.hi*d>=n*scale,'outward enclosure');
 assert.equal(Q.sign(value),n<0n?-1:n>0n?1:0);
 assert.equal(Q.floor(value),n/d-(n<0n&&n%d!==0n?1n:0n));values.push(value);
}
const restored=createExactDag({leafBits:32}),map=restored.importGraph(graph.exportGraph());
for(const value of values){const v=restored.decodeImported(graph.encode(value),map);
 assert.equal(restored.digest(v),graph.digest(value));assert.equal(restored.Q.sign(v),Q.sign(value));}
assert(graph.stats().certified_decisions>0);assert.equal(graph.stats().interval_values_in_state,0);
assert(restored.exportGraph().imported_proof_sources.length>0);
'''
    subprocess.run(["node", "-e", script], input=json.dumps(rows), text=True,
                   capture_output=True, cwd=ROOT, check=True)


def test_exact_boundaries_and_linear_identity():
    script = r'''
const assert=require('node:assert/strict');
const {createExactDag}=require('./experiments/grvq_ansatz_comparison/exact_dag.cjs');
const graph=createExactDag({leafBits:32}),Q=graph.Q;
const tiny=Q.mk(1n,1n<<500n),a=Q.add(Q.ONE,tiny),b=Q.sub(Q.ONE,tiny);
assert.equal(Q.floor(a),1n);assert.equal(Q.floor(b),0n);
assert.equal(Q.cmp(a,Q.ONE),1);assert.equal(Q.cmp(b,Q.ONE),-1);
const x=Q.parse('12345678901234567890123456789/7000000000000000001');
const y=Q.parse('-88997766554433221100/1700000000000000003');
const mapped=graph.linearMap([[1n,2n],[-2n,-4n]],3n,[x,y]);
assert(Q.isZero(Q.add(Q.mul(Q.TWO,mapped[0]),mapped[1])));
assert(graph.certificates.some(c=>c.proof==='exact_rational_linear_identity'));
assert(graph.stats().maximum_certificate_bits>=512);
assert.throws(()=>Q.div(Q.ONE,Q.ZERO),/Division by zero/);
assert.throws(()=>Q.parse('0.1'),/Invalid exact rational/);
'''
    subprocess.run(["node", "-e", script], text=True, capture_output=True,
                   cwd=ROOT, check=True)
