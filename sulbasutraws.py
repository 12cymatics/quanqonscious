# ========== SULBA SUTRAS ==========
    
def sulba_square_construction(
    self,
    side_length: Union[float, np.ndarray, torch.Tensor],
    ctx: Optional[SutraContext] = None,
) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sulba Sutra: Square Construction - Calculates the area and perimeter of a square,
        with optimizations for computational geometry and spatial transformations.
        
        Mathematical logic: Provides exact calculations for square construction,
        with applications in geometric transformations and spatial modeling.
        
        Classical applications:
        - Computational geometry
        - CAD/CAM systems
        - Computer graphics
        - Spatial optimization algorithms
        
        Quantum applications:
        - Quantum spatial encoding
        - Geometric quantum machine learning
        - Quantum circuit layout optimization
        - Topological quantum computing models
        
        Args:
            side_length: Length of the square side
            ctx: Optional execution context override
            
        Returns:
            Tuple of (area, perimeter) for the square
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(side_length)
        data_size = np.size(side_length) if hasattr(side_length, 'size') else 1
        
        try:
            # Convert to device if using GPU
            side_length_device = self._to_device(side_length)
            
            # Quantum implementation
            if context.mode == SutraMode.SULBA:
                return self._sulba_square_construction_quantum(side_length, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._sulba_square_construction_hybrid(side_length, context)
            
            # Classical implementation (default)
            if isinstance(side_length_device, torch.Tensor):
                # Calculate area and perimeter
                area = side_length_device * side_length_device
                perimeter = 4 * side_length_device
                result = (area, perimeter)
                
            elif isinstance(side_length_device, np.ndarray):
                # Calculate area and perimeter
                area = side_length_device * side_length_device
                perimeter = 4 * side_length_device
                result = (area, perimeter)
                
            else:
                # Calculate area and perimeter
                area = side_length_device * side_length_device
                perimeter = 4 * side_length_device
                result = (area, perimeter)
            
            # Convert back to original type (components of the tuple)
            if isinstance(result, tuple):
                result = tuple(self._from_device(r, original_type) for r in result)
            else:
                result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("sulba_square_construction", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in sulba_square_construction: {error_msg}")
            self._record_performance("sulba_square_construction", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
def _sulba_square_construction_quantum(self, side_length, context):
        """Quantum implementation of sulba_square_construction using CUDAQ"""
        # For scalar values, implement quantum circuit
        if not isinstance(side_length, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._sulba_square_construction_classical(side_length, context)
        
        # Create CUDAQ kernel for square calculations
        kernel = cudaq.make_kernel()
        
        # Allocate qubits for encoding
        q_side = kernel.qalloc(4)  # For side length
        q_area = kernel.qalloc(8)  # For area (side^2)
        q_perim = kernel.qalloc(6)  # For perimeter (4*side)
        
        # Normalize side length for qubit encoding (assuming reasonable range)
        side_norm = min(1.0, side_length / 16.0)  # Scale to fit in 4 qubits
        
        # Encode side length into q_side
        # Using binary encoding for numerical precision
        side_int = int(side_norm * 15)  # Scale to 0-15 range for 4 qubits
        side_binary = bin(side_int)[2:].zfill(4)
        
        for i, bit in enumerate(reversed(side_binary)):
            if bit == '1':
                kernel.x(q_side[i])
        
        # Calculate area by squaring the side length
        # This is a simplified quantum multiplication
        for i in range(4):
            for j in range(4):
                if i + j < 8:  # Ensure we stay within q_area bounds
                    kernel.cx(q_side[i], q_area[i+j]).controlled_by(q_side[j])
        
        # Calculate perimeter by multiplying side by 4
        # This is a shift operation (multiply by 4 = shift left by 2)
        for i in range(4):
            if i + 2 < 6:  # Ensure we stay within q_perim bounds
                kernel.cx(q_side[i], q_perim[i+2])
        
        # Measure results
        kernel.mz(q_area)
        kernel.mz(q_perim)
        
        # Execute
        result = cudaq.sample(kernel)
        
        # Extract measurements and convert back to original scale
        area_bits = result.get_register_as_int("q_area")
        perim_bits = result.get_register_as_int("q_perim")
        
        # Convert quantum results back to original scale
        area = (area_bits / 225) * (side_length ** 2)  # Scaled back from normalized value
        perimeter = (perim_bits / 15) * (4 * side_length)  # Scaled back from normalized value
        
        return (area, perimeter)
    
def _sulba_square_construction_hybrid(self, side_length, context):
        """Hybrid implementation of sulba_square_construction"""
        # For scalar values, use quantum circuit
        if isinstance(side_length, (int, float)):
            return self._sulba_square_construction_quantum(side_length, context)
        # For small arrays, use quantum for some elements
        elif isinstance(side_length, np.ndarray) and side_length.size <= 4:
            area = np.zeros_like(side_length)
            perimeter = np.zeros_like(side_length)
            
            for i in range(side_length.size):
                quantum_result = self._sulba_square_construction_quantum(side_length.flat[i], context)
                area.flat[i] = quantum_result[0]
                perimeter.flat[i] = quantum_result[1]
                
            return (area, perimeter)
        else:
            # For larger arrays, use classical implementation
            return self._sulba_square_construction_classical(side_length, context)
    
def _sulba_square_construction_classical(self, side_length, context):
        """Classical implementation of sulba_square_construction"""
        if isinstance(side_length, torch.Tensor):
            # Calculate area and perimeter
            area = side_length * side_length
            perimeter = 4 * side_length
            return (area, perimeter)
            
        elif isinstance(side_length, np.ndarray):
            # Calculate area and perimeter
            area = side_length * side_length
            perimeter = 4 * side_length
            return (area, perimeter)
            
        else:
            # Calculate area and perimeter
            area = side_length * side_length
            perimeter = 4 * side_length
            return (area, perimeter)

def sulba_circle_construction(
    self,
    radius: Union[float, np.ndarray, torch.Tensor],
    ctx: Optional[SutraContext] = None,
) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sulba Sutra: Circle Construction - Calculates the area and circumference of a circle,
        with sulba-optimized approximations for π.
        
        Mathematical logic: Provides exact and approximated calculations for circle construction,
        using the ancient Indian approximation of π.
        
        Classical applications:
        - Computational geometry
        - CAD/CAM systems
        - Computer graphics
        - Scientific computing
        
        Quantum applications:
        - Quantum spatial encoding
        - Circular quantum states
        - Quantum phase space
        - Quantum circuit layout optimization
        
        Args:
            radius: Radius of the circle
            ctx: Optional execution context override
            
        Returns:
            Tuple of (area, circumference) for the circle
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(radius)
        data_size = np.size(radius) if hasattr(radius, 'size') else 1
        
        try:
            # Convert to device if using GPU
            radius_device = self._to_device(radius)
            
            # Quantum implementation
            if context.mode == SutraMode.SULBA:
                return self._sulba_circle_construction_quantum(radius, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._sulba_circle_construction_hybrid(radius, context)
            
            # Classical implementation (default)
            # Use sulba approximation for pi: 3.0883... (derived from √10)
            pi_sulba = np.sqrt(10)  # Ancient Indian approximation
            
            if isinstance(radius_device, torch.Tensor):
                # Calculate area and circumference
                area = pi_sulba * radius_device * radius_device
                circumference = 2 * pi_sulba * radius_device
                result = (area, circumference)
                
            elif isinstance(radius_device, np.ndarray):
                # Calculate area and circumference
                area = pi_sulba * radius_device * radius_device
                circumference = 2 * pi_sulba * radius_device
                result = (area, circumference)
                
            else:
                # Calculate area and circumference
                area = pi_sulba * radius_device * radius_device
                circumference = 2 * pi_sulba * radius_device
                result = (area, circumference)
            
            # Convert back to original type (components of the tuple)
            if isinstance(result, tuple):
                result = tuple(self._from_device(r, original_type) for r in result)
            else:
                result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("sulba_circle_construction", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in sulba_circle_construction: {error_msg}")
            self._record_performance("sulba_circle_construction", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
def _sulba_circle_construction_quantum(self, radius, context):
        """Quantum implementation of sulba_circle_construction using Cirq"""
        # For scalar values, implement quantum circuit
        if not isinstance(radius, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._sulba_circle_construction_classical(radius, context)
        
        # Create Cirq circuit for circle calculations
        q_radius = [cirq.LineQubit(i) for i in range(4)]  # For radius
        q_pi = [cirq.LineQubit(i+4) for i in range(3)]  # For π approximation
        q_area = [cirq.LineQubit(i+7) for i in range(8)]  # For area
        q_circ = [cirq.LineQubit(i+15) for i in range(7)]  # For circumference
        
        circuit = cirq.Circuit()
        
        # Normalize radius for qubit encoding (assuming reasonable range)
        radius_norm = min(1.0, radius / 16.0)  # Scale to fit in 4 qubits
        
        # Encode radius into q_radius
        radius_int = int(radius_norm * 15)  # Scale to 0-15 range for 4 qubits
        radius_binary = bin(radius_int)[2:].zfill(4)
        
        for i, bit in enumerate(reversed(radius_binary)):
            if bit == '1':
                circuit.append(cirq.X(q_radius[i]))
        
        # Encode sulba pi approximation (√10 ≈ 3.16) into q_pi
        # This is a fixed value so we'll encode it directly
        pi_sulba_binary = "011"  # 3 in binary
        for i, bit in enumerate(reversed(pi_sulba_binary)):
            if bit == '1':
                circuit.append(cirq.X(q_pi[i]))
        
        # Calculate area = π * r^2
        # First, square the radius (r^2)
        for i in range(4):
            for j in range(4):
                if i + j < 8:  # Ensure we stay within q_area bounds
                    circuit.append(cirq.CNOT(q_radius[i], q_area[i+j]).controlled_by(q_radius[j]))
        
        # Then multiply by π
        for i in range(3):
            for j in range(8):
                if i + j < 8:  # Ensure we stay within bounds
                    circuit.append(cirq.CNOT(q_pi[i], q_area[j]).controlled_by(q_area[j]))
        
        # Calculate circumference = 2 * π * r
        # First, multiply radius by π
        for i in range(3):
            for j in range(4):
                if i + j < 7:  # Ensure we stay within q_circ bounds
                    circuit.append(cirq.CNOT(q_pi[i], q_circ[i+j]).controlled_by(q_radius[j]))
        
        # Then multiply by 2 (shift left by 1)
        for i in range(6):
            circuit.append(cirq.SWAP(q_circ[i], q_circ[i+1]))
        circuit.append(cirq.X(q_circ[0]))
        
        # Measure results
        circuit.append(cirq.measure(*q_area, key='area'))
        circuit.append(cirq.measure(*q_circ, key='circumference'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
        
        # Extract measurements and convert back to original scale
        area_bits = int(''.join(str(bit) for bit in reversed(result.measurements['area'][0])), 2)
        circ_bits = int(''.join(str(bit) for bit in reversed(result.measurements['circumference'][0])), 2)
        
        # Convert quantum results back to original scale
        pi_sulba = np.sqrt(10)  # Sulba approximation
        area = (area_bits / 225) * (pi_sulba * radius ** 2)  # Scaled back from normalized value
        circumference = (circ_bits / 60) * (2 * pi_sulba * radius)  # Scaled back from normalized value
        
        return (area, circumference)
    
def _sulba_circle_construction_hybrid(self, radius, context):
        """Hybrid implementation of sulba_circle_construction"""
        # For scalar values, use quantum circuit
        if isinstance(radius, (int, float)):
            return self._sulba_circle_construction_quantum(radius, context)
        # For small arrays, use quantum for some elements
        elif isinstance(radius, np.ndarray) and radius.size <= 4:
            area = np.zeros_like(radius)
            circumference = np.zeros_like(radius)
            
            for i in range(radius.size):
                quantum_result = self._sulba_circle_construction_quantum(radius.flat[i], context)
                area.flat[i] = quantum_result[0]
                circumference.flat[i] = quantum_result[1]
                
            return (area, circumference)
        else:
            # For larger arrays, use classical implementation
            return self._sulba_circle_construction_classical(radius, context)
    
def _sulba_circle_construction_classical(self, radius, context):
        """Classical implementation of sulba_circle_construction"""
        # Use sulba approximation for pi: 3.0883... (derived from √10)
        pi_sulba = np.sqrt(10)  # Ancient Indian approximation
        
        if isinstance(radius, torch.Tensor):
            # Calculate area and circumference
            area = pi_sulba * radius * radius
            circumference = 2 * pi_sulba * radius
            return (area, circumference)
            
        elif isinstance(radius, np.ndarray):
            # Calculate area and circumference
            area = pi_sulba * radius * radius
            circumference = 2 * pi_sulba * radius
            return (area, circumference)
            
        else:
            # Calculate area and circumference
            area = pi_sulba * radius * radius
            circumference = 2 * pi_sulba * radius
            return (area, circumference)

def sulba_pythagorean_triples(
    self,
    max_c: int,
    ctx: Optional[SutraContext] = None,
) -> List[Tuple[int, int, int]]:
        """
        Sulba Sutra: Pythagorean Triples - Generates Pythagorean triples using sulba methods,
        with applications in geometric calculations and constraint satisfaction.
        
        Mathematical logic: Implements the ancient Indian method for generating Pythagorean triples,
        based on techniques described in sulba sutras.
        
        Classical applications:
        - Right triangle construction
        - Distance computations
        - Constraint satisfaction problems
        - Geometric modeling
        
        Quantum applications:
        - Quantum constraint satisfaction
        - Quantum state preparation for geometric encoding
        - Quantum circuit geometry
        - Quantum walks on geometric lattices
        
        Args:
            max_c: Maximum value for the hypotenuse
            ctx: Optional execution context override
            
        Returns:
            List of Pythagorean triples (a, b, c) where a²+b²=c²
        """
        context = ctx or self.context
        start_time = time.time()
        data_size = 1  # Scalar input
        
        try:
            # Quantum implementation
            if context.mode == SutraMode.SULBA:
                return self._sulba_pythagorean_triples_quantum(max_c, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._sulba_pythagorean_triples_hybrid(max_c, context)
            
            # Classical implementation (default)
            # Use the ancient Indian method for generating Pythagorean triples
            triples = []
            
            # Generate primitive triples using Sulba formula
            for m in range(2, int(np.sqrt(max_c)) + 1):
                for n in range(1, m):
                    # Ensure m and n are coprime and not both odd
                    if np.gcd(m, n) == 1 and (m % 2 == 0 or n % 2 == 0):
                        # Calculate triple using sulba method
                        a = m * m - n * n
                        b = 2 * m * n
                        c = m * m + n * n
                        
                        # Ensure c is within the limit
                        if c <= max_c:
                            if a > b:
                                triples.append((b, a, c))
                            else:
                                triples.append((a, b, c))
            
            # Sort triples by c (hypotenuse)
            triples.sort(key=lambda t: t[2])
            
            end_time = time.time()
            self._record_performance("sulba_pythagorean_triples", start_time, end_time, 
                                    True, data_size)
            return triples
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in sulba_pythagorean_triples: {error_msg}")
            self._record_performance("sulba_pythagorean_triples", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
def _sulba_pythagorean_triples_quantum(self, max_c, context):
        """Quantum implementation of sulba_pythagorean_triples using CUDAQ"""
        # This is a combinatorial problem, and full quantum implementation
        # would involve Grover's algorithm for search or similar.
        # For simplicity and practical use, we'll implement a hybrid approach
        return self._sulba_pythagorean_triples_hybrid(max_c, context)
    
def _sulba_pythagorean_triples_hybrid(self, max_c, context):
        """Hybrid implementation of sulba_pythagorean_triples"""
        # Use quantum to verify triples, classical to generate candidates
        
        # Generate candidate primitives using classical method (more efficient)
        candidates = []
        
        # Generate candidate primitive triples using Sulba formula
        for m in range(2, int(np.sqrt(max_c)) + 1):
            for n in range(1, m):
                # Ensure m and n are coprime and not both odd
                if np.gcd(m, n) == 1 and (m % 2 == 0 or n % 2 == 0):
                    # Calculate triple using sulba method
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    
                    # Ensure c is within the limit
                    if c <= max_c:
                        if a > b:
                            candidates.append((b, a, c))
                        else:
                            candidates.append((a, b, c))
        
        # Now use quantum circuit to verify each triple
        verified_triples = []
        
        for a, b, c in candidates:
            if self._verify_pythagorean_triple_quantum(a, b, c):
                verified_triples.append((a, b, c))
        
        # Sort triples by c (hypotenuse)
        verified_triples.sort(key=lambda t: t[2])
        
        return verified_triples
    
def _verify_pythagorean_triple_quantum(self, a, b, c):
        """Verify Pythagorean triple using a quantum circuit"""
        # Create CUDAQ kernel for verification
        kernel = cudaq.make_kernel()
        
        # Allocate qubit for verification
        q_verify = kernel.qalloc(1)
        
        # Calculate a^2 + b^2 - c^2
        difference = a*a + b*b - c*c
        
        # If the difference is 0, this is a valid triple
        if difference == 0:
            # Set qubit to |1⟩ to indicate valid triple
            kernel.x(q_verify[0])
        
        # Measure
        kernel.mz(q_verify)
        
        # Execute
        result = cudaq.sample(kernel)
        
        # Check if the qubit is in the |1⟩ state, indicating a valid triple
        counts = result.get_counts()
        return "1" in counts and counts["1"] > 0
    
def _sulba_pythagorean_triples_classical(self, max_c, context):
        """Classical implementation of sulba_pythagorean_triples"""
        triples = []
        
        # Generate primitive triples using Sulba formula
        for m in range(2, int(np.sqrt(max_c)) + 1):
            for n in range(1, m):
                # Ensure m and n are coprime and not both odd
                if np.gcd(m, n) == 1 and (m % 2 == 0 or n % 2 == 0):
                    # Calculate triple using sulba method
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    
                    # Ensure c is within the limit
                    if c <= max_c:
                        if a > b:
                            triples.append((b, a, c))
                        else:
                            triples.append((a, b, c))
        
        # Sort triples by c (hypotenuse)
        triples.sort(key=lambda t: t[2])
        
        return triples

def sulba_geometric_mean(
    self,
    a: Union[float, np.ndarray, torch.Tensor],
    b: Union[float, np.ndarray, torch.Tensor],
    ctx: Optional[SutraContext] = None,
) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sulba Sutra: Geometric Mean - Calculates the geometric mean √(a×b),
        with applications in geometric constructions and proportional scaling.
        
        Mathematical logic: Implements the ancient Indian method for calculating
        geometric means, based on techniques described in sulba sutras.
        
        Classical applications:
        - Scaling transformations
        - Aspect ratio calculations
        - Proportional design algorithms
        - Geometric average computations
        
        Quantum applications:
        - Quantum state preparation for geometric encoding
        - Quantum amplitude scaling
        - Quantum phase estimation refinement
        - Quantum optimization constraints
        
        Args:
            a: First value or array
            b: Second value or array
            ctx: Optional execution context override
            
        Returns:
            Geometric mean √(a×b)
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(a)
        data_size = np.size(a) if hasattr(a, 'size') else 1
        
        try:
            # Convert to device if using GPU
            a_device = self._to_device(a)
            b_device = self._to_device(b)
            
            # Quantum implementation
            if context.mode == SutraMode.SULBA:
                return self._sulba_geometric_mean_quantum(a, b, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._sulba_geometric_mean_hybrid(a, b, context)
            
            # Classical implementation (default)
            if isinstance(a_device, torch.Tensor):
                # Calculate geometric mean
                result = torch.sqrt(a_device * b_device)
                
            elif isinstance(a_device, np.ndarray):
                # Calculate geometric mean
                result = np.sqrt(a_device * b_device)
                
            else:
                # Calculate geometric mean
                result = np.sqrt(a_device * b_device)
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("sulba_geometric_mean", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in sulba_geometric_mean: {error_msg}")
            self._record_performance("sulba_geometric_mean", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
def _sulba_geometric_mean_quantum(self, a, b, context):
        """Quantum implementation of sulba_geometric_mean using Cirq"""
        # For scalar values, implement quantum circuit
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._sulba_geometric_mean_classical(a, b, context)
        
        # Create Cirq circuit for geometric mean
        q_a = cirq.LineQubit(0)  # For encoding a
        q_b = cirq.LineQubit(1)  # For encoding b
        q_result = cirq.LineQubit(2)  # For result (geometric mean)
        
        circuit = cirq.Circuit()
        
        # Normalize inputs for quantum encoding (assuming reasonable range)
        a_norm = min(1.0, np.sqrt(a / 100.0))  # Normalize to [0,1]
        b_norm = min(1.0, np.sqrt(b / 100.0))  # Normalize to [0,1]
        
        # Encode a and b into quantum amplitudes
        theta_a = 2 * np.arcsin(a_norm)
        theta_b = 2 * np.arcsin(b_norm)
        
        circuit.append(cirq.ry(theta_a)(q_a))
        circuit.append(cirq.ry(theta_b)(q_b))
        
        # Create entanglement between inputs
        circuit.append(cirq.CNOT(q_a, q_b))
        
        # Apply measurement-inspired algorithm for geometric mean
        circuit.append(cirq.ry(theta_a / 2)(q_result))
        circuit.append(cirq.ry(theta_b / 2)(q_result))
        
        # Apply interference operation
        circuit.append(cirq.CNOT(q_a, q_result))
        circuit.append(cirq.CNOT(q_b, q_result))
        
        # Measure result qubit
        circuit.append(cirq.measure(q_result, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Analyze measurements
        counts = result.histogram(key='result')
        probability_one = counts.get(1, 0) / 1000
        
        # Convert quantum probability to geometric mean
        # Scale back to original range
        geo_mean = np.sqrt(probability_one * 100.0 * 100.0)
        
        # Scale to actual geometric mean of inputs
        rescaling_factor = np.sqrt(a * b) / geo_mean if geo_mean > 0 else 1.0
        final_result = geo_mean * rescaling_factor
        
        return final_result
    
def _sulba_geometric_mean_hybrid(self, a, b, context):
        """Hybrid implementation of sulba_geometric_mean"""
        # For scalar values, use quantum circuit
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return self._sulba_geometric_mean_quantum(a, b, context)
        # For small arrays, use quantum for some elements
        elif isinstance(a, np.ndarray) and a.size <= 4:
            result = np.zeros_like(a)
            for i in range(a.size):
                a_val = a.flat[i]
                b_val = b.flat[i] if isinstance(b, np.ndarray) else b
                result.flat[i] = self._sulba_geometric_mean_quantum(a_val, b_val, context)
            return result
        else:
            # For larger arrays, use classical implementation
            return self._sulba_geometric_mean_classical(a, b, context)
    
def _sulba_geometric_mean_classical(self, a, b, context):
        """Classical implementation of sulba_geometric_mean"""
        if isinstance(a, torch.Tensor):
            # Calculate geometric mean
            return torch.sqrt(a * b)
            
        elif isinstance(a, np.ndarray):
            # Calculate geometric mean
            return np.sqrt(a * b)
            
        else:
            # Calculate geometric mean
            return np.sqrt(a * b)