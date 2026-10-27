import itertools
import json

def cube_neighbors(vertex, dimensions):
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError('Positive integer dimension required')
    if type(vertex) is not int or not 0 <= vertex < 1 << dimensions:
        raise ValueError('Vertex outside the hypercube')
    return tuple(vertex ^ (1 << axis) for axis in range(dimensions))

def cube_route(source, target, dimensions):
    cube_neighbors(source, dimensions)
    cube_neighbors(target, dimensions)
    path = [source]
    for axis in range(dimensions):
        if (path[-1] ^ target) & (1 << axis):
            path.append(path[-1] ^ (1 << axis))
    return tuple(path)

def periodic_slots(coordinate, shape):
    if len(coordinate) != len(shape) or not shape:
        raise ValueError('Coordinate and shape dimensions differ')
    if any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('Axis lengths must be positive integers')
    if any(type(x) is not int or not 0 <= x < n for x, n in zip(coordinate, shape)):
        raise ValueError('Coordinate outside periodic grid')
    slots = []
    for axis, size in enumerate(shape):
        for direction in (-1, 1):
            neighbor = list(coordinate)
            neighbor[axis] = (neighbor[axis] + direction) % size
            slots.append((axis, direction, tuple(neighbor)))
    return tuple(slots)

def gray(index):
    if type(index) is not int or index < 0:
        raise ValueError('Nonnegative integer Gray index required')
    return index ^ (index >> 1)

def inverse_gray(value):
    if type(value) is not int or value < 0:
        raise ValueError('Nonnegative integer Gray value required')
    result = 0
    while value:
        result ^= value
        value >>= 1
    return result

def lattice_slots(coordinate, shape, periodic_axes):
    periodic_slots(coordinate, shape)
    if len(periodic_axes) != len(shape) or any(type(p) is not bool for p in periodic_axes):
        raise ValueError('One explicit periodic boolean per axis required')
    slots = []
    for axis, size in enumerate(shape):
        for direction in (-1, 1):
            neighbor = list(coordinate)
            candidate = neighbor[axis] + direction
            if periodic_axes[axis]:
                candidate %= size
            elif not 0 <= candidate < size:
                continue
            neighbor[axis] = candidate
            slots.append((axis, direction, tuple(neighbor)))
    return tuple(slots)

def inverse_morton2(value, bits):
    if type(bits) is not int or bits < 1:
        raise ValueError('Positive integer bit count required')
    if type(value) is not int or not 0 <= value < 1 << (2 * bits):
        raise ValueError('Morton key outside domain')
    return tuple(sum(((value >> (2 * bit + axis)) & 1) << bit for bit in range(bits)) for axis in (0, 1))

def bfs_distance(source, target, dimensions):
    frontier = {source}
    visited = {source}
    distance = 0
    while target not in frontier:
        frontier = {w for v in frontier for w in cube_neighbors(v, dimensions)} - visited
        visited |= frontier
        distance += 1
        if not frontier:
            raise ValueError('Disconnected vertices')
    return distance

def morton2(x, y, bits):
    if type(bits) is not int or bits < 1:
        raise ValueError('Positive integer bit count required')
    if any(type(v) is not int or not 0 <= v < 1 << bits for v in (x, y)):
        raise ValueError('Coordinate outside Morton domain')
    value = 0
    for bit in range(bits):
        value |= ((x >> bit) & 1) << (2 * bit)
        value |= ((y >> bit) & 1) << (2 * bit + 1)
    return value

def verify():
    results = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        results.append(name)
    vertices = range(16)
    edges = {tuple(sorted((v, w))) for v in vertices for w in cube_neighbors(v, 4)}
    check('Q4 has 16 vertices and 32 simple edges', len(edges) == 32)
    check('Q4 has four distinct unsigned neighbors', all(len(set(cube_neighbors(v, 4))) == 4 for v in vertices))
    check('Q4 adjacency is symmetric', all(v in cube_neighbors(w, 4) for v in vertices for w in cube_neighbors(v, 4)))
    check('Q4 routes retain their requested endpoints', all(cube_route(a, b, 4)[0] == a and cube_route(a, b, 4)[-1] == b for a in vertices for b in vertices))
    check('Q4 shortest route length equals Hamming distance', all(len(cube_route(a, b, 4)) - 1 == bin(a ^ b).count('1') for a in vertices for b in vertices))
    check('Q4 routes use valid edges', all(all(y in cube_neighbors(x, 4) for x, y in zip(route, route[1:])) for a in vertices for b in vertices for route in [cube_route(a, b, 4)]))
    for mask in vertices:
        f = lambda v: -1 if bin(v & mask).count('1') % 2 else 1
        check('Q4 negative Laplacian Walsh mask ' + str(mask), all(sum(f(w) - f(v) for w in cube_neighbors(v, 4)) == -2 * bin(mask).count('1') * f(v) for v in vertices))
    order = [gray(i) for i in vertices]
    check('Gray order is bijective on Q4', len(set(order)) == 16 and set(order) == set(vertices))
    check('Gray consecutive vertices differ in one bit', all(bin(a ^ b).count('1') == 1 for a, b in zip(order, order[1:])))
    shape = (3, 5)
    coordinates = tuple(itertools.product(*(range(n) for n in shape)))
    check('3 by 5 toroidal grid has four distinct neighbors', all(len({n for _, _, n in periodic_slots(v, shape)}) == 4 for v in coordinates))
    check('Fully periodic first-axis negative seam wraps correctly', (0, -1, (2, 0)) in periodic_slots((0, 0), shape))
    slots = periodic_slots((0, 0, 0, 0), (2, 2, 2, 2))
    check('Length-two periodic axes retain eight stencil slots', len(slots) == 8)
    check('Those slots have four unique neighbors', len({n for _, _, n in slots}) == 4)
    encode = lambda c: sum(x << axis for axis, x in enumerate(c))
    weighted = sum((-1 if encode(n) & 1 else 1) - 1 for _, _, n in slots)
    check('Duplicated periodic stencil is not the unsigned Q4 Laplacian', weighted == -4 and weighted != -2)
    points = tuple(itertools.product(range(4), repeat=2))
    codes = {morton2(x, y, 2) for x, y in points}
    check('2D Morton map is bijective on the tested 4 by 4 grid', codes == set(range(16)))
    ordered = sorted(points, key=lambda p: morton2(*p, 2))
    check('Morton consecutive addresses need not be nearest neighbors', any(abs(a[0]-b[0]) + abs(a[1]-b[1]) > 1 for a, b in zip(ordered, ordered[1:])))
    check('Independent BFS distances equal Q4 Hamming distances', all(bfs_distance(a, b, 4) == bin(a ^ b).count('1') for a in vertices for b in vertices))
    check('Gray closing pair differs in one bit', bin(order[-1] ^ order[0]).count('1') == 1)
    check('Gray inverse exhaustive round trips through eight bits', all(inverse_gray(gray(i)) == i and gray(inverse_gray(i)) == i for i in range(256)))
    check('Length-one periodic axis retains two self stencil slots', periodic_slots((0,), (1,)) == ((0, -1, (0,)), (0, 1, (0,))))
    check('Length-one simple graph excludes self loops', not {n for _, _, n in periodic_slots((0,), (1,)) if n != (0,)})
    check('Mixed finite-periodic adjacency does not wrap finite axis', lattice_slots((0, 0), (3, 3), (False, True)) == ((0, 1, (1, 0)), (1, -1, (0, 2)), (1, 1, (0, 1))))
    check('Finite adjacency corner has only in-domain neighbors', lattice_slots((0, 0), (3, 3), (False, False)) == ((0, 1, (1, 0)), (1, 1, (0, 1))))
    check('Mixed periodic slots preserve inverse edges', all((axis, -direction, v) in lattice_slots(n, (3, 2), (False, True)) for v in itertools.product(range(3), range(2)) for axis, direction, n in lattice_slots(v, (3, 2), (False, True))))
    check('Morton exhaustive inverse round trips on 32 by 32 grid', all(inverse_morton2(morton2(x, y, 5), 5) == (x, y) for x, y in itertools.product(range(32), repeat=2)))
    check('Morton key inverse round trips through ten bits', all(morton2(*inverse_morton2(key, 5), 5) == key for key in range(1024)))
    check('Morton awkward rectangle remains an injective sparse key map', len({morton2(x, y, 3) for x, y in itertools.product(range(3), range(5))}) == 15)
    rejected = [
        ('cube zero dimensions', lambda: cube_neighbors(0, 0)),
        ('cube negative vertex', lambda: cube_neighbors(-1, 4)),
        ('cube upper vertex bound', lambda: cube_neighbors(16, 4)),
        ('cube bool dimension', lambda: cube_neighbors(0, True)),
        ('Gray negative input', lambda: gray(-1)),
        ('Gray bool input', lambda: gray(True)),
        ('inverse Gray negative input', lambda: inverse_gray(-1)),
        ('Morton coordinate bound', lambda: morton2(4, 0, 2)),
        ('Morton invalid bit count', lambda: morton2(0, 0, 0)),
        ('Morton key overflow', lambda: inverse_morton2(16, 2)),
        ('periodic zero extent', lambda: periodic_slots((0,), (0,))),
        ('periodic invalid coordinate', lambda: periodic_slots((3,), (3,))),
        ('periodic dimension mismatch', lambda: periodic_slots((0,), (3, 3))),
        ('mixed missing periodic declaration', lambda: lattice_slots((0,), (3,), ())),
        ('mixed nonboolean declaration', lambda: lattice_slots((0,), (3,), (1,)))
    ]
    for name, operation in rejected:
        try:
            operation()
        except ValueError:
            check('Reject ' + name, True)
        else:
            check('Reject ' + name, False)
    return {'structural_checks_passed': len(results), 'checks': results, 'performance_benchmark': 'not_run', 'production_kernel_validation': 'not_performed', 'physical_model_validation': 'not_performed'}

if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
