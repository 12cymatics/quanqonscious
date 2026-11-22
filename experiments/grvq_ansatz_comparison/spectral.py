"""Exact spectral experiments consuming the repository's Vedic compiler.

The operator is a newly specified dimensionless benchmark on [0,1]^d.  Its
weights are experimental choices, not recovered GRVQ physical constants.
The source mass, derivative, triple-product, endpoint and reciprocal-symbol
results determine the generalised eigenproblem A c = E M c.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

import sympy as sp


def _q(value):
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("An exact rational value is required")
    if isinstance(value, sp.Rational):
        return Fraction(int(value.p), int(value.q))
    return Fraction(value)


def _matrix(values, size=4):
    if len(values) != size * size:
        raise ValueError("A matrix has the wrong number of exact entries")
    return tuple(tuple(_q(values[i*size+j]) for j in range(size)) for i in range(size))


def _expr(coefficients, x):
    return sum(sp.Rational(c) * x**k for k, c in enumerate(coefficients))


def _integral(expression, x):
    return _q(sp.integrate(sp.expand(expression), (x, 0, 1)))


def _flat(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _flat(item)
    else:
        yield value


def _compare(name, actual, expected, report):
    actual, expected = tuple(_flat(actual)), tuple(_flat(expected))
    if len(actual) != len(expected):
        raise ArithmeticError(f"{name}: incompatible reference size")
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        if sp.simplify(sp.sympify(observed)-sp.sympify(reference)) != 0:
            raise ArithmeticError(f"{name}[{index}]: {observed} != {reference}")
    report.append({"quantity": name, "entries": len(actual), "equal": len(actual)})


def load_and_verify_polynomials(source):
    data = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else source
    if data.get("schema") != "grvq-vedic-polynomial-data/v1":
        raise ValueError("Unexpected exact polynomial data schema")
    nodes = tuple(_q(q) for q in data["nodes"])
    if len(nodes) != 4 or len(set(nodes)) != 4:
        raise ValueError("Four distinct cubic interpolation nodes are required")
    x, z = sp.symbols("x z")
    basis = tuple(sp.expand(sp.prod((x-sp.Rational(nodes[j])) /
                  (sp.Rational(nodes[i])-sp.Rational(nodes[j]))
                  for j in range(4) if j != i)) for i in range(4))
    first, second = tuple(sp.diff(p,x) for p in basis), tuple(sp.diff(p,x,2) for p in basis)
    report = []
    for name, expected in (("basis", basis), ("first_derivatives", first), ("second_derivatives", second)):
        _compare(name, tuple(_expr(row,x) for row in data[name]), expected, report)
    matrices = {}
    for name, functions in (("mass",basis), ("stiffness",first), ("bending",second)):
        direct = tuple(_integral(a*b,x) for a in functions for b in functions)
        _compare(name, tuple(map(_q,data[name])), direct, report)
        matrices[name] = _matrix(data[name])
    direct_triple = tuple(_integral(a*b*c,x) for a in basis for b in basis for c in basis)
    _compare("triple", tuple(map(_q,data["triple"])), direct_triple, report)
    _compare("mean", tuple(map(_q,data["mean"])), tuple(_integral(p,x) for p in basis), report)
    for side, point in (("left",0),("right",1)):
        endpoint = tuple(_q(p.subs(x,point)) for p in basis)
        _compare("endpoint_"+side, tuple(map(_q,data["endpoints"][side])), endpoint, report)
        _compare("endpoint_mass_"+side, tuple(map(_q,data["endpoint_mass_"+side])),
                 tuple(a*b for a in endpoint for b in endpoint), report)
    reconstructed_mass = [[Fraction(0) for _ in range(4)] for _ in range(4)]
    for factor in data["mass_factors"]:
        weight = _q(factor["weight"])
        if weight <= 0:
            raise ArithmeticError("A source mass-factor weight is not positive")
        coefficients = tuple(map(_q,factor["coefficients"]))
        for i in range(4):
            for j in range(4):
                reconstructed_mass[i][j] += weight*coefficients[i]*coefficients[j]
    _compare("mass_factor_reconstruction", reconstructed_mass, matrices["mass"], report)
    leading_minors = tuple(_q(sp.Matrix(matrices["mass"])[:k,:k].det()) for k in range(1,5))
    if any(value <= 0 for value in leading_minors):
        raise ArithmeticError("The mass matrix is not positive definite")
    _compare("shape_samples", [[_q(q) for q in row] for row in data["shape_samples"]],
             [[_q(p.subs(x,sp.Rational(point))) for p in basis] for point in data["shape_points"]],report)
    a,b,c,xy,yz,zx = tuple(map(_q,data["potential_coefficients"]))
    direct_potential = tuple(a*u*u+b*v*v+c*w*w+xy*u*v+yz*v*w+zx*w*u
                             for u in nodes for v in nodes for w in nodes)
    _compare("potential_3d", tuple(map(_q,data["potential_3d"])),direct_potential,report)
    symbol = data["reciprocal_symbol"]
    weights, coefficients = tuple(map(_q,symbol["input_weights"])),tuple(map(_q,symbol["coefficients"]))
    if len(weights)%2 != 1 or weights != weights[::-1]:
        raise ValueError("The kinetic stencil must be odd-length and reciprocal")
    radius = len(weights)//2
    converted = sp.expand(z**radius * sum(sp.Rational(q)*(z+1/z)**k for k,q in enumerate(coefficients)))
    original = sum(sp.Rational(w)*z**k for k,w in enumerate(weights))
    _compare("reciprocal_symbol_identity", converted, original, report)
    _compare("reciprocal_symbol_endpoints", tuple(map(_q,symbol["values_at_minus_two_and_two"])),
             tuple(sum(q*t**k for k,q in enumerate(coefficients)) for t in (-2,2)),report)
    if len(data["methods"]) != 29 or any(method["calls"] < 1 for method in data["methods"]):
        raise ArithmeticError("The compiled source did not execute all twenty-nine mapped methods")
    result = {"schema":"grvq-polynomial-reference-checks-v1", "comparisons":report,
              "entries_compared":sum(row["entries"] for row in report),
              "all_equal":True, "mass_leading_minors":tuple(map(str,leading_minors)),
              "reference":"Independent SymPy cardinal products, symbolic derivatives and exact integrals",
              "source_hashes":data["sources"]}
    return data, result


def _identity(n):
    return tuple(tuple(Fraction(i==j) for j in range(n)) for i in range(n))


def _transpose(matrix):
    return tuple(zip(*matrix))


def _matmul(a,b):
    columns = _transpose(b)
    return tuple(tuple(sum((x*y for x,y in zip(row,column)),Fraction(0))
                       for column in columns) for row in a)


def _tensor_entry(factors, left, right):
    value = Fraction(1)
    for matrix,i,j in zip(factors,left,right):
        value *= matrix[i][j]
    return value


def _symbol_matrix(data):
    n = 4
    adjacency = tuple(tuple(Fraction((j==(i+1)%n)+(j==(i-1)%n)) for j in range(n)) for i in range(n))
    identity = _identity(n)
    coefficients = tuple(map(_q,data["reciprocal_symbol"]["coefficients"]))
    result = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    power = identity
    for coefficient in coefficients:
        for i in range(n):
            for j in range(n):
                result[i][j] += coefficient*power[i][j]
        power = _matmul(power,adjacency)
    result = tuple(tuple(row) for row in result)
    weights = tuple(map(_q,data["reciprocal_symbol"]["input_weights"]))
    radius = len(weights)//2
    direct = tuple(tuple(sum((w for k,w in enumerate(weights) if j==(i+k-radius)%n),Fraction(0))
                         for j in range(n)) for i in range(n))
    if result != direct:
        raise ArithmeticError("Reciprocal polynomial and direct cyclic stencil matrices differ")
    return result


def build_spectral_problem(data, dimension=2, *, bend_weight=Fraction(1,100),
                           boundary_weight=Fraction(1,10), periodic_weight=Fraction(1,100),
                           potential_weight=Fraction(1)):
    if dimension not in (2,3):
        raise ValueError("This cubic tensor experiment supports dimensions two and three")
    weights = {"bend":_q(bend_weight),"boundary":_q(boundary_weight),
               "periodic":_q(periodic_weight),"potential":_q(potential_weight)}
    if any(value < 0 for value in weights.values()):
        raise ValueError("This positive-energy benchmark requires nonnegative term weights")
    nodes = tuple(map(_q,data["nodes"]))
    mass, kinetic, bending = (_matrix(data[key]) for key in ("mass","stiffness","bending"))
    endpoints = tuple(tuple(map(_q,data["endpoints"][side])) for side in ("left","right"))
    boundary = tuple(tuple(sum(v[i]*v[j] for v in endpoints) for j in range(4)) for i in range(4))
    stored_boundary = tuple(tuple(_q(data["endpoint_mass_left"][4*i+j])+
                                  _q(data["endpoint_mass_right"][4*i+j]) for j in range(4)) for i in range(4))
    if boundary != stored_boundary:
        raise ArithmeticError("Consumed nodal endpoint weights disagree with endpoint matrices")
    cyclic = _symbol_matrix(data)
    periodic = _matmul(_matmul(_transpose(cyclic),mass),cyclic)
    indices = tuple(itertools.product(range(4),repeat=dimension))
    triples = tuple(map(_q,data["triple"]))
    nodal_potential = tuple(_q(data["potential_3d"][(k[0]*4+k[1])*4+(k[2] if dimension==3 else 0)])
                            for k in indices)
    potential, M, A = [], [], []
    for left in indices:
        potential_row, mass_row, operator_row = [], [], []
        for right in indices:
            mij = _tensor_entry((mass,)*dimension,left,right)
            vij = Fraction(0)
            for k, node_value in zip(indices,nodal_potential):
                term = node_value
                for axis in range(dimension):
                    term *= triples[(left[axis]*4+right[axis])*4+k[axis]]
                vij += term
            value = weights["potential"]*vij
            for axis in range(dimension):
                for one_dimension,weight in ((kinetic,Fraction(1,2)),(bending,weights["bend"]),
                                             (boundary,weights["boundary"]),(periodic,weights["periodic"])):
                    factors = tuple(one_dimension if d==axis else mass for d in range(dimension))
                    value += weight*_tensor_entry(factors,left,right)
            potential_row.append(vij); mass_row.append(mij); operator_row.append(value)
        potential.append(tuple(potential_row)); M.append(tuple(mass_row)); A.append(tuple(operator_row))
    potential, M, A = tuple(potential),tuple(M),tuple(A)
    if A != _transpose(A) or M != _transpose(M):
        raise ArithmeticError("The exact spectral operator or mass matrix is asymmetric")
    x = sp.Symbol("x")
    basis = tuple(sp.expand(sp.prod((x-sp.Rational(nodes[j])) /
                  (sp.Rational(nodes[i])-sp.Rational(nodes[j]))
                  for j in range(4) if j!=i)) for i in range(4))
    moments = {degree:tuple(tuple(_integral(x**degree*a*b,x) for b in basis) for a in basis)
               for degree in range(3)}
    coefficients = tuple(map(_q,data["potential_coefficients"]))
    direct_potential = []
    for left in indices:
        row = []
        for right in indices:
            value = sum((coefficients[d]*_tensor_entry(tuple(moments[2] if axis==d else moments[0]
                         for axis in range(dimension)),left,right) for d in range(dimension)),Fraction(0))
            for coefficient,axis_a,axis_b in ((coefficients[3],0,1),(coefficients[4],1,2),(coefficients[5],2,0)):
                if axis_a < dimension and axis_b < dimension:
                    value += coefficient*_tensor_entry(tuple(moments[1] if axis in (axis_a,axis_b) else moments[0]
                                                for axis in range(dimension)),left,right)
            row.append(value)
        direct_potential.append(tuple(row))
    if potential != tuple(direct_potential):
        raise ArithmeticError("Source-triple contraction differs from direct weighted potential integrals")
    digest = hashlib.sha256(json.dumps({"A":[[str(v) for v in row] for row in A],
                                        "M":[[str(v) for v in row] for row in M]},sort_keys=True).encode()).hexdigest()
    return {"name":f"vedic_spectral_{dimension}d", "dimension":dimension,"nqubits":2*dimension,
            "A":A,"M":M,"potential_matrix":potential,"potential_values":nodal_potential,
            "coordinates":tuple(tuple(nodes[k] for k in index) for index in indices),
            "metadata":{"domain":f"[0,1]^{dimension}","basis":"tensor cubic cardinal polynomials",
                        "coefficient_count":4**dimension,"index_order":"last coordinate varies fastest",
                        "weights":{key:str(value) for key,value in weights.items()},"kinetic_weight":"1/2",
                        "boundary":"finite endpoint energy penalty, not an imposed Dirichlet constraint",
                        "periodic_term":"L.T M1 L with L=p(C+C.T); C is the four-point cyclic shift",
                        "potential":"quadratic nodal samples from retainedQuadratic contracted with all source triple products",
                        "interpretation":"new explicit dimensionless spectral benchmark, not a recovered complete GRVQ field equation",
                        "exact_matrix_sha256":digest,
                        "verification":{"potential_matrix_entries":16**dimension,
                                        "potential_all_equal":True,"cyclic_stencil_entries":16,
                                        "cyclic_stencil_all_equal":True,"symmetric":True},
                        "methods":data["methods"],"sources":data["sources"]}}


def build_spectral_suite(source):
    data, verification = load_and_verify_polynomials(source)
    return tuple(build_spectral_problem(data,dimension) for dimension in (2,3)), verification
