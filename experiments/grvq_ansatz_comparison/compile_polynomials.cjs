"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const crypto = require("node:crypto");
const { performance } = require("node:perf_hooks");

const root = path.resolve(__dirname, "../..");
const sourcePath = path.join(root, "dns/vedic-polynomial.js");
const operatorPath = path.join(root, "dns/vedic-operators.js");
const source = fs.readFileSync(sourcePath, "utf8");
const operators = fs.readFileSync(operatorPath, "utf8");
const argument = process.argv.find(value => value.startsWith("--classical-ids="));
const classicalIds = argument ? argument.slice("--classical-ids=".length).split(",").map(Number) : [];
if (classicalIds.some(id => !Number.isInteger(id) || id < 1 || id > 29)) {
  throw new Error("Classical comparison ids must lie between one and twenty-nine");
}
const names = [
  "ekadhikena", "nikhilam", "urdhva", "paravartya", "shunyamCoefficient",
  "nodalWeights", "symmetricTranslate", "purana", "calana", "yavadunam",
  "vyashti", "sesanyankena", "sopantyaProductDerivative", "ekanyunena",
  "gunitasamuccaya", "gunakasamuccaya", "anurupyena", "shishyateKnownLinear",
  "adyamDivide", "kevalaih", "osculateEvaluate", "tavadunamCube",
  "extendedSquare", "antyayordasakepi", "antyayorevaProductSecond",
  "coefficientSum", "retainedQuadratic", "reciprocalPolynomial",
  "gunitasamuccayaSamuccayagunitah"
];
const counts = Object.fromEntries(names.map(name => [name, 0]));
let instrumented = source;
for (const name of names) {
  const pattern = new RegExp(`function ${name}\\(([^)]*)\\)\\{`);
  if (!pattern.test(instrumented)) throw new Error(`Source function is absent: ${name}`);
  instrumented = instrumented.replace(pattern, `$&__record(${JSON.stringify(name)});`);
}
if (classicalIds.length) {
  const { overrideSource } = require("./conventional_overrides.cjs");
  const marker = "const api=";
  if (!instrumented.includes(marker)) throw new Error("Source API boundary changed");
  instrumented = instrumented.replace(marker, overrideSource(classicalIds) + "\n" + marker);
}
const sandbox = {
  module: { exports: {} },
  __record(name) { counts[name]++; },
};
vm.runInNewContext(instrumented, sandbox, { filename: sourcePath });
const V = sandbox.module.exports;
const operatorSandbox = { module: { exports: {} }, require(name) {
  if (name !== "./vedic-polynomial.js") throw new Error(`Unexpected dependency: ${name}`);
  return V;
}};
vm.runInNewContext(operators, operatorSandbox, { filename: operatorPath });
const O = operatorSandbox.module.exports;
const q = (n, d = 1) => V.Q(BigInt(n), BigInt(d));
const stringify = value => JSON.stringify(value, (_, x) => {
  if (x && typeof x.n === "bigint" && typeof x.d === "bigint") return V.qstr(x);
  if (typeof x === "bigint") return x.toString();
  return x;
}, 2);
const begin = performance.now();
const nodes = [q(0), q(1, 3), q(2, 3), q(1)];
const moment = O.compileMoments(nodes, q(0), q(1));
const first = V.lagrangeDerivativeBasis(nodes, 1);
const stiffness = first.flatMap(a => first.map(b => V.integralProduct([a, b], q(0), q(1))));
const shapePoints = Array.from({ length: 65 }, (_, i) => q(i, 64));
const shapeSamples = shapePoints.map(x => moment.exact.basis.map(p => V.osculateEvaluate(p, x)));
const quadraticCoefficients = [q(1), q(2), q(3), q(1, 2), q(1, 3), q(1, 4)];
const quadraticPotential = nodes.flatMap(x => nodes.flatMap(y => nodes.map(z =>
  V.retainedQuadratic(quadraticCoefficients, [x, y, z]))));
const symbol = O.compileSymmetricSymbol([q(1), q(-2), q(1)]);
const symbolAtEndpoints = [q(-2), q(2)].map(t => V.osculateEvaluate(symbol.coefficients, t));
const endpoints = {
  left: V.nodalWeights(nodes, q(0)),
  right: V.nodalWeights(nodes, q(1)),
};
const elapsed = performance.now() - begin;
const output = {
  schema: "grvq-vedic-polynomial-data/v1",
  classical_comparison_ids: classicalIds,
  sources: [sourcePath, operatorPath].map(p => ({
    path: path.relative(root, p),
    sha256: crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex")
  })),
  nodes,
  basis: moment.exact.basis,
  first_derivatives: first,
  second_derivatives: moment.exact.secondDerivative,
  mass: moment.exact.mass,
  stiffness,
  bending: moment.exact.bendingMass,
  triple: moment.exact.flux,
  mean: moment.exact.mean,
  mass_factors: moment.exact.massFactors,
  endpoint_mass_left: moment.exact.endpointMassLeft,
  endpoint_mass_right: moment.exact.endpointMassRight,
  endpoints,
  shape_points: shapePoints,
  shape_samples: shapeSamples,
  potential_coefficients: quadraticCoefficients,
  potential_3d: quadraticPotential,
  reciprocal_symbol: {
    input_weights: symbol.weights,
    coefficients: symbol.coefficients,
    values_at_minus_two_and_two: symbolAtEndpoints
  },
  methods: names.map((name, i) => ({
    id: i + 1, name: V.names[i], implementation: name, calls: counts[name],
    interpretation: classicalIds.includes(i+1) ? "independent conventional exact control" : "repository polynomial extension or book arithmetic application"
  })),
  generation_ms: elapsed,
  tracing: "Function-entry counters inserted into the unchanged source bodies; all returned coefficients are exact rational values."
};
if (!classicalIds.length && output.methods.some(method => method.calls === 0)) {
  throw new Error(`Unexecuted methods: ${output.methods.filter(method => !method.calls).map(method => method.id)}`);
}
process.stdout.write(stringify(output) + "\n");
