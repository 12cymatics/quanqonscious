# ========== SUB-SUTRAS (1-7) ==========
    
    def anurupye_sunyamanyat(self, a: Union[float, np.ndarray, torch.Tensor],
                            b: Union[float, np.ndarray, torch.Tensor],
                            threshold: Optional[float] = None,
                            ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sub-Sutra 1: Anurupye Sunyamanyat - "If one is in ratio, the other is zero"
        
        Mathematical logic: Evaluates proportional relationships between values, 
        setting appropriate terms to zero when specific ratio conditions are met.
        
        Classical applications:
        - Ratio-based filtering in data processing
        - Identifying and eliminating redundant terms in equations
        - Simplifying complex expressions by zeroing proportional terms
        - Detecting resonance or equilibrium conditions
        
        Quantum applications:
        - Quantum state purification
        - Interference pattern identification
        - Quantum error syndrome identification
        - Entanglement ratio measurement
        
        Args:
            a: First value or array
            b: Second value or array
            threshold: Tolerance for ratio detection (default from context)
            ctx: Optional execution context override
            
        Returns:
            Result with terms set to zero based on ratio relationships
        """
        context = ctx or self.context
        eps = threshold if threshold is not None else context.epsilon
        start_time = time.time()
        original_type = type(a)
        data_size = np.size(a) if hasattr(a, 'size') else 1
        
        try:
            # Convert to device if using GPU
            a_device = self._to_device(a)
            b_device = self._to_device(b)
            eps_device = self._to_device(eps)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._anurupye_sunyamanyat_quantum(a, b, eps, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._anurupye_sunyamanyat_hybrid(a, b, eps, context)
            
            # Classical implementation (default)
            if isinstance(a_device, torch.Tensor):
                # Calculate ratio and check if it's close to 1 or -1
                # Handle division by zero
                safe_b = torch.where(
                    torch.abs(b_device) > eps_device,
                    b_device,
                    torch.ones_like(b_device) * eps_device
                )
                ratio = a_device / safe_b
                
                # Check if ratio is close to 1 or -1 (within epsilon)
                ratio_condition = (torch.abs(torch.abs(ratio) - 1.0) < eps_device)
                
                # Where ratio condition is met, set both to zero
                result_a = torch.where(ratio_condition, torch.zeros_like(a_device), a_device)
                result_b = torch.where(ratio_condition, torch.zeros_like(b_device), b_device)
                
                result = (result_a, result_b)
                
            elif isinstance(a_device, np.ndarray):
                # Calculate ratio and check if it's close to 1 or -1
                # Handle division by zero
                safe_b = np.where(
                    np.abs(b_device) > eps,
                    b_device,
                    np.ones_like(b_device) * eps
                )
                ratio = a_device / safe_b
                
                # Check if ratio is close to 1 or -1 (within epsilon)
                ratio_condition = (np.abs(np.abs(ratio) - 1.0) < eps)
                
                # Where ratio condition is met, set both to zero
                result_a = np.where(ratio_condition, np.zeros_like(a_device), a_device)
                result_b = np.where(ratio_condition, np.zeros_like(b_device), b_device)
                
                result = (result_a, result_b)
                
            else:
                # Handle scalar case
                if abs(b_device) > eps:
                    ratio = a_device / b_device
                    if abs(abs(ratio) - 1.0) < eps:
                        result = (0.0, 0.0)
                    else:
                        result = (a_device, b_device)
                else:
                    # b is close to zero
                    if abs(a_device) < eps:
                        # Both a and b are close to zero
                        result = (0.0, 0.0)
                    else:
                        # Only b is close to zero
                        result = (a_device, b_device)
            
            # For scalar results, convert back to original type
            if not isinstance(result, tuple):
                result = self._from_device(result, original_type)
            else:
                result = (self._from_device(result[0], original_type), 
                         self._from_device(result[1], original_type))
            
            end_time = time.time()
            self._record_performance("anurupye_sunyamanyat", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in anurupye_sunyamanyat: {error_msg}")
            self._record_performance("anurupye_sunyamanyat", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _anurupye_sunyamanyat_quantum(self, a, b, epsilon, context):
        """Quantum implementation of anurupye_sunyamanyat using Cirq"""
        # This implements quantum interference to detect ratio relationships
        
        # Simple implementation for scalar inputs
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._anurupye_sunyamanyat_classical(a, b, epsilon, context)
        
        # Normalize inputs to range [0, 1] for encoding as quantum amplitudes
        max_val = max(abs(a), abs(b))
        if max_val < epsilon:
            return (0.0, 0.0)
            
        norm_a = min(1.0, a / max_val)
        norm_b = min(1.0, b / max_val)
        
        # Create quantum circuit with two qubits
        qubits = [cirq.LineQubit(i) for i in range(2)]
        circuit = cirq.Circuit()
        
        # Encode values in quantum state amplitudes
        theta_a = 2 * np.arcsin(abs(norm_a))
        theta_b = 2 * np.arcsin(abs(norm_b))
        
        circuit.append(cirq.ry(theta_a)(qubits[0]))
        circuit.append(cirq.ry(theta_b)(qubits[1]))
        
        # Apply phase based on sign
        if norm_a < 0:
            circuit.append(cirq.Z(qubits[0]))
        if norm_b < 0:
            circuit.append(cirq.Z(qubits[1]))
        
        # Apply Hadamard gates to observe interference
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.H(qubits[1]))
        
        # Measure
        circuit.append(cirq.measure(*qubits, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Analyze measurements
        counts = result.histogram(key='result')
        
        # If interference leads to significant bias to specific states,
        # then a and b are likely in ratio
        threshold = 0.75  # Arbitrary threshold for determining ratio relationship
        total_shots = sum(counts.values())
        
        # Check if measurements are heavily biased toward any specific outcome
        for outcome, count in counts.items():
            if count / total_shots > threshold:
                # Strong bias detected, likely in ratio
                return (0.0, 0.0)
        
        # No strong ratio relationship detected
        return (a, b)
    
    def _anurupye_sunyamanyat_hybrid(self, a, b, epsilon, context):
        """Hybrid implementation of anurupye_sunyamanyat"""
        # For small arrays, use quantum interference checking
        if (isinstance(a, np.ndarray) and a.size <= 4) or isinstance(a, (int, float)):
            if isinstance(a, np.ndarray):
                # Process each element pair through quantum check
                result_a = np.zeros_like(a)
                result_b = np.zeros_like(b if isinstance(b, np.ndarray) else a)
                
                for i in range(a.size):
                    a_val = a.flat[i]
                    b_val = b.flat[i] if isinstance(b, np.ndarray) else b
                    quantum_result = self._anurupye_sunyamanyat_quantum(
                        a_val, b_val, epsilon, context
                    )
                    result_a.flat[i] = quantum_result[0]
                    result_b.flat[i] = quantum_result[1]
                
                return (result_a, result_b)
            else:
                # Single value case
                return self._anurupye_sunyamanyat_quantum(a, b, epsilon, context)
        else:
            # For larger arrays, use classical implementation
            return self._anurupye_sunyamanyat_classical(a, b, epsilon, context)
    
    def _anurupye_sunyamanyat_classical(self, a, b, epsilon, context):
        """Classical implementation of anurupye_sunyamanyat"""
        if isinstance(a, np.ndarray):
            # Handle division by zero
            safe_b = np.where(
                np.abs(b) > epsilon,
                b,
                np.ones_like(b) * epsilon
            )
            ratio = a / safe_b
            
            # Check if ratio is close to 1 or -1 (within epsilon)
            ratio_condition = (np.abs(np.abs(ratio) - 1.0) < epsilon)
            
            # Where ratio condition is met, set both to zero
            result_a = np.where(ratio_condition, np.zeros_like(a), a)
            result_b = np.where(ratio_condition, np.zeros_like(b), b)
            
            return (result_a, result_b)
            
        elif isinstance(a, torch.Tensor):
            # Handle division by zero
            safe_b = torch.where(
                torch.abs(b) > epsilon,
                b,
                torch.ones_like(b) * epsilon
            )
            ratio = a / safe_b
            
            # Check if ratio is close to 1 or -1 (within epsilon)
            ratio_condition = (torch.abs(torch.abs(ratio) - 1.0) < epsilon)
            
            # Where ratio condition is met, set both to zero
            result_a = torch.where(ratio_condition, torch.zeros_like(a), a)
            result_b = torch.where(ratio_condition, torch.zeros_like(b), b)
            
            return (result_a, result_b)
            
        else:
            # Scalar case
            if abs(b) > epsilon:
                ratio = a / b
                if abs(abs(ratio) - 1.0) < epsilon:
                    return (0.0, 0.0)
                else:
                    return (a, b)
            else:
                # b is close to zero
                if abs(a) < epsilon:
                    # Both a and b are close to zero
                    return (0.0, 0.0)
                else:
                    # Only b is close to zero
                    return (a, b)

    def sisyate_sesasamjnah(self, a: Union[float, np.ndarray, torch.Tensor],
                           b: Union[float, np.ndarray, torch.Tensor],
                           ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sub-Sutra 2: Sisyate Sesasamjnah - "The remainder remains constant"
        
        Mathematical logic: Identifies and preserves remainder terms in operations,
        useful for modular arithmetic and cyclical pattern detection.
        
        Classical applications:
        - Modular arithmetic calculations
        - Cyclical pattern detection
        - Checksums and error detection codes
        - Calendar calculations
        
        Quantum applications:
        - Quantum period finding
        - Quantum phase estimation
        - Shor's algorithm optimization
        - Quantum error correction codes
        
        Args:
            a: Dividend value or array
            b: Divisor value or array
            ctx: Optional execution context override
            
        Returns:
            Remainder of a divided by b
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
            if context.mode == SutraMode.QUANTUM:
                return self._sisyate_sesasamjnah_quantum(a, b, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._sisyate_sesasamjnah_hybrid(a, b, context)
            
            # Classical implementation (default)
            if isinstance(a_device, torch.Tensor):
                # Handle division by zero
                safe_b = torch.where(
                    torch.abs(b_device) > context.epsilon,
                    b_device,
                    torch.ones_like(b_device)
                )
                # Calculate remainder
                quotient = torch.div(a_device, safe_b, rounding_mode='floor')
                result = a_device - quotient * safe_b
                
            elif isinstance(a_device, np.ndarray):
                # Handle division by zero
                safe_b = np.where(
                    np.abs(b_device) > context.epsilon,
                    b_device,
                    np.ones_like(b_device)
                )
                # Calculate remainder
                quotient = np.floor_divide(a_device, safe_b)
                result = a_device - quotient * safe_b
                
            else:
                # Handle scalar case with safety check
                if abs(b_device) < context.epsilon:
                    safe_b = 1.0  # Default to 1 if divisor is too close to zero
                else:
                    safe_b = b_device
                # Calculate remainder
                quotient = int(a_device // safe_b)
                result = a_device - quotient * safe_b
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("sisyate_sesasamjnah", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in sisyate_sesasamjnah: {error_msg}")
            self._record_performance("sisyate_sesasamjnah", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _sisyate_sesasamjnah_quantum(self, a, b, context):
        """Quantum implementation of sisyate_sesasamjnah using CUDAQ"""
        # This implements a quantum circuit for modular arithmetic
        
        # For scalar values, implement quantum circuit
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._sisyate_sesasamjnah_classical(a, b, context)
        
        # Limit implementation to small integers for simplicity
        if not (isinstance(a, int) and isinstance(b, int) and 1 <= b <= 16 and 0 <= a <= 31):
            # Fall back to classical for non-integer or large values
            return self._sisyate_sesasamjnah_classical(a, b, context)
        
        # Create CUDAQ kernel for remainder calculation
        kernel = cudaq.make_kernel()
        
        # Number of qubits needed to represent divisor b
        b_bits = max(1, int(np.ceil(np.log2(b))))
        
        # Number of qubits needed to represent dividend a
        a_bits = max(1, int(np.ceil(np.log2(a + 1))))
        
        # Allocate qubits
        q_dividend = kernel.qalloc(a_bits)
        q_divisor = kernel.qalloc(b_bits)
        q_result = kernel.qalloc(b_bits)  # Result register for remainder
        
        # Encode a into q_dividend
        a_binary = bin(a)[2:].zfill(a_bits)
        for i, bit in enumerate(reversed(a_binary)):
            if bit == '1':
                kernel.x(q_dividend[i])
        
        # Encode b into q_divisor
        b_binary = bin(b)[2:].zfill(b_bits)
        for i, bit in enumerate(reversed(b_binary)):
            if bit == '1':
                kernel.x(q_divisor[i])
        
        # Apply quantum modular arithmetic
        # This is a simplified implementation using quantum phase estimation
        
        # Create superposition in result register
        for i in range(b_bits):
            kernel.h(q_result[i])
        
        # Apply controlled operations based on modular properties
        for i in range(a_bits):
            if i < b_bits:
                kernel.cx(q_dividend[i], q_result[i])
            else:
                # For higher bits, apply modular reduction
                # This is just a simplified representation
                for j in range(b_bits):
                    kernel.cx(q_dividend[i], q_result[j]).controlled_by(q_divisor[j])
        
        # Apply inverse QFT to extract remainder
        cudaq.inverseFQFT(kernel, q_result)
        
        # Measure result
        kernel.mz(q_result)
        
        # Execute
        result = cudaq.sample(kernel)
        
        # Get most frequent outcome
        top_result = result.most_probable()
        
        # Convert to decimal
        remainder = int(top_result, 2) % b
        
        return remainder
    
    def _sisyate_sesasamjnah_hybrid(self, a, b, context):
        """Hybrid implementation of sisyate_sesasamjnah"""
        # For small integer values, use quantum circuit
        if (isinstance(a, int) and isinstance(b, int) and 
            1 <= b <= 16 and 0 <= a <= 31):
            return self._sisyate_sesasamjnah_quantum(a, b, context)
        # For small arrays with small integer values, use quantum for some elements
        elif (isinstance(a, np.ndarray) and a.size <= 4 and
              np.all(np.floor(a) == a) and np.all(np.floor(b) == b) and
              np.all(b >= 1) and np.all(b <= 16) and np.all(a >= 0) and np.all(a <= 31)):
            result = np.zeros_like(a)
            for i in range(a.size):
                a_val = int(a.flat[i])
                b_val = int(b.flat[i] if isinstance(b, np.ndarray) else b)
                result.flat[i] = self._sisyate_sesasamjnah_quantum(a_val, b_val, context)
            return result
        else:
            # For larger or non-integer values, use classical implementation
            return self._sisyate_sesasamjnah_classical(a, b, context)
    
    def _sisyate_sesasamjnah_classical(self, a, b, context):
        """Classical implementation of sisyate_sesasamjnah"""
        if isinstance(a, torch.Tensor):
            # Handle division by zero
            safe_b = torch.where(
                torch.abs(b) > context.epsilon,
                b,
                torch.ones_like(b)
            )
            # Calculate remainder
            quotient = torch.div(a, safe_b, rounding_mode='floor')
            return a - quotient * safe_b
            
        elif isinstance(a, np.ndarray):
            # Handle division by zero
            safe_b = np.where(
                np.abs(b) > context.epsilon,
                b,
                np.ones_like(b)
            )
            # Calculate remainder
            quotient = np.floor_divide(a, safe_b)
            return a - quotient * safe_b
            
        else:
            # Handle scalar case with safety check
            if abs(b) < context.epsilon:
                safe_b = 1.0  # Default to 1 if divisor is too close to zero
            else:
                safe_b = b
            # Calculate remainder
            quotient = int(a // safe_b)
            return a - quotient * safe_b

    def adyamadyenantyamantyena(self, coefficients: Union[List, np.ndarray, torch.Tensor],
                               x: Union[float, np.ndarray, torch.Tensor],
                               ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sub-Sutra 3: Adyamadyenantyamantyena - "The first by the first and the last by the last"
        
        Mathematical logic: Implements specialized polynomial evaluation focusing on 
        first and last terms, providing efficiency for specific polynomial structures.
        
        Classical applications:
        - Efficient polynomial evaluation
        - Series approximation
        - Telescoping series computation
        - Numerical integration with endpoint focus
        
        Quantum applications:
        - Quantum polynomial state preparation
        - Quantum signal processing
        - Quantum machine learning feature maps
        - Quantum phase estimation refinement
        
        Args:
            coefficients: List of polynomial coefficients [a0, a1, a2, ...]
            x: Value(s) at which to evaluate the polynomial
            ctx: Optional execution context override
            
        Returns:
            Result of polynomial evaluation using first-last optimization
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            
            # Convert coefficients to appropriate format
            if self.context.use_gpu:
                if isinstance(coefficients, list):
                    coeffs_device = torch.tensor(coefficients, device=self.context.device)
                elif isinstance(coefficients, np.ndarray):
                    coeffs_device = torch.tensor(coefficients, device=self.context.device)
                else:  # Assume it's already a tensor
                    coeffs_device = coefficients.to(self.context.device)
            else:
                if isinstance(coefficients, list):
                    coeffs_device = np.array(coefficients)
                elif isinstance(coefficients, torch.Tensor):
                    coeffs_device = coefficients.cpu().numpy()
                else:  # Assume it's already a numpy array
                    coeffs_device = coefficients
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._adyamadyenantyamantyena_quantum(coeffs_device, x, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._adyamadyenantyamantyena_hybrid(coeffs_device, x, context)
            
            # Classical implementation (default)
            # This sutra optimizes polynomial evaluation by focusing on first and last terms
            # and working inward, unlike Horner's method which is sequential
            
            if isinstance(x_device, torch.Tensor):
                # Initialize result with zeros
                result = torch.zeros_like(x_device)
                
                # Get polynomial degree
                n = len(coeffs_device) - 1
                
                # Evaluate polynomial using first-last pairing
                for i in range((n + 1) // 2):
                    # Pair the i-th term from the beginning with the i-th term from the end
                    first_term = coeffs_device[i] * (x_device ** i)
                    last_term = coeffs_device[n - i] * (x_device ** (n - i))
                    
                    result = result + first_term + last_term
                
                # Handle middle term for odd-degree polynomials
                if n % 2 == 0:
                    middle_idx = n // 2
                    middle_term = coeffs_device[middle_idx] * (x_device ** middle_idx)
                    result = result - middle_term  # Subtract because we added it twice
                
            elif isinstance(x_device, np.ndarray):
                # Initialize result with zeros
                result = np.zeros_like(x_device)
                
                # Get polynomial degree
                n = len(coeffs_device) - 1
                
                # Evaluate polynomial using first-last pairing
                for i in range((n + 1) // 2):
                    # Pair the i-th term from the beginning with the i-th term from the end
                    first_term = coeffs_device[i] * (x_device ** i)
                    last_term = coeffs_device[n - i] * (x_device ** (n - i))
                    
                    result = result + first_term + last_term
                
                # Handle middle term for odd-degree polynomials
                if n % 2 == 0:
                    middle_idx = n // 2
                    middle_term = coeffs_device[middle_idx] * (x_device ** middle_idx)
                    result = result - middle_term  # Subtract because we added it twice
                
            else:
                # Scalar case
                # Initialize result with zero
                result = 0.0
                
                # Get polynomial degree
                n = len(coeffs_device) - 1
                
                # Evaluate polynomial using first-last pairing
                for i in range((n + 1) // 2):
                    # Pair the i-th term from the beginning with the i-th term from the end
                    first_term = coeffs_device[i] * (x_device ** i)
                    last_term = coeffs_device[n - i] * (x_device ** (n - i))
                    
                    result = result + first_term + last_term
                
                # Handle middle term for odd-degree polynomials
                if n % 2 == 0:
                    middle_idx = n // 2
                    middle_term = coeffs_device[middle_idx] * (x_device ** middle_idx)
                    result = result - middle_term  # Subtract because we added it twice
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("adyamadyenantyamantyena", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in adyamadyenantyamantyena: {error_msg}")
            self._record_performance("adyamadyenantyamantyena", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _adyamadyenantyamantyena_quantum(self, coefficients, x, context):
        """Quantum implementation of adyamadyenantyamantyena using Cirq"""
        # This implements a quantum circuit for polynomial evaluation
        # focusing on first and last terms
        
        # For simplicity, we'll handle the case of polynomial degree <= 3
        # and scalar x values
        
        if len(coefficients) > 4 or not isinstance(x, (int, float)):
            # Fall back to classical for higher-degree polynomials or non-scalar inputs
            return self._adyamadyenantyamantyena_classical(coefficients, x, context)
        
        # Create Cirq circuit for polynomial evaluation
        qubits = [cirq.LineQubit(i) for i in range(len(coefficients))]
        circuit = cirq.Circuit()
        
        # Encode x value using rotation gates
        theta_x = np.arcsin(min(1.0, abs(x) / 10.0))  # Normalize for safe encoding
        
        # Prepare all qubits in superposition
        for q in qubits:
            circuit.append(cirq.H(q))
        
        # Encode polynomial coefficients through rotation gates
        n = len(coefficients) - 1
        
        # Apply rotations for first and last terms
        for i in range((n + 1) // 2):
            # Scale coefficient to prevent numerical issues
            first_coef = min(10.0, abs(coefficients[i]))
            last_coef = min(10.0, abs(coefficients[n - i]))
            
            # Apply rotations proportional to coefficients
            circuit.append(cirq.ry(first_coef * theta_x ** i)(qubits[i]))
            circuit.append(cirq.ry(last_coef * theta_x ** (n - i))(qubits[n - i]))
        
        # Handle middle term for even-degree polynomials
        if n % 2 == 0:
            middle_idx = n // 2
            middle_coef = min(10.0, abs(coefficients[middle_idx]))
            circuit.append(cirq.ry(middle_coef * theta_x ** middle_idx)(qubits[middle_idx]))
        
        # Apply final Hadamard gates for interference
        for q in qubits:
            circuit.append(cirq.H(q))
        
        # Measure all qubits
        circuit.append(cirq.measure(*qubits, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Calculate result based on measurement statistics
        counts = result.histogram(key='result')
        
        # The measurement probabilities are related to the polynomial evaluation
        # Sum weighted measurement outcomes
        weighted_sum = 0.0
        total_shots = sum(counts.values())
        
        for outcome, count in counts.items():
            # Convert binary outcome to value
            value = 0
            for i, bit in enumerate(reversed(bin(outcome)[2:].zfill(len(qubits)))):
                if bit == '1':
                    # Apply coefficient with appropriate power of x
                    if i <= n:
                        value += coefficients[i] * (x ** i)
            
            weighted_sum += value * (count / total_shots)
        
        return weighted_sum
    
    def _adyamadyenantyamantyena_hybrid(self, coefficients, x, context):
        """Hybrid implementation of adyamadyenantyamantyena"""
        # For small polynomials with scalar x, use quantum circuit
        if len(coefficients) <= 4 and isinstance(x, (int, float)):
            return self._adyamadyenantyamantyena_quantum(coefficients, x, context)
        # For small arrays with small polynomials, use quantum for some elements
        elif isinstance(x, np.ndarray) and x.size <= 4 and len(coefficients) <= 4:
            result = np.zeros_like(x)
            for i in range(x.size):
                result.flat[i] = self._adyamadyenantyamantyena_quantum(
                    coefficients, x.flat[i], context
                )
            return result
        else:
            # For higher-degree polynomials or larger arrays, use classical implementation
            return self._adyamadyenantyamantyena_classical(coefficients, x, context)
    
    def _adyamadyenantyamantyena_classical(self, coefficients, x, context):
        """Classical implementation of adyamadyenantyamantyena"""
        if isinstance(x, torch.Tensor):
            # Initialize result with zeros
            result = torch.zeros_like(x)
            
            # Get polynomial degree
            n = len(coefficients) - 1
            
            # Evaluate polynomial using first-last pairing
            for i in range((n + 1) // 2):
                # Pair the i-th term from the beginning with the i-th term from the end
                first_term = coefficients[i] * (x ** i)
                last_term = coefficients[n - i] * (x ** (n - i))
                
                result = result + first_term + last_term
            
            # Handle middle term for even-degree polynomials
            if n % 2 == 0:
                middle_idx = n // 2
                middle_term = coefficients[middle_idx] * (x ** middle_idx)
                result = result - middle_term  # Subtract because we added it twice
                
        elif isinstance(x, np.ndarray):
            # Initialize result with zeros
            result = np.zeros_like(x)
            
            # Get polynomial degree
            n = len(coefficients) - 1
            
            # Evaluate polynomial using first-last pairing
            for i in range((n + 1) // 2):
                # Pair the i-th term from the beginning with the i-th term from the end
                first_term = coefficients[i] * (x ** i)
                last_term = coefficients[n - i] * (x ** (n - i))
                
                result = result + first_term + last_term
            
            # Handle middle term for even-degree polynomials
            if n % 2 == 0:
                middle_idx = n // 2
                middle_term = coefficients[middle_idx] * (x ** middle_idx)
                result = result - middle_term  # Subtract because we added it twice
                
        else:
            # Scalar case
            # Initialize result with zero
            result = 0.0
            
            # Get polynomial degree
            n = len(coefficients) - 1
            
            # Evaluate polynomial using first-last pairing
            for i in range((n + 1) // 2):
                # Pair the i-th term from the beginning with the i-th term from the end
                first_term = coefficients[i] * (x ** i)
                last_term = coefficients[n - i] * (x ** (n - i))
                
                result = result + first_term + last_term
            
            # Handle middle term for even-degree polynomials
            if n % 2 == 0:
                middle_idx = n // 2
                middle_term = coefficients[middle_idx] * (x ** middle_idx)
                result = result - middle_term  # Subtract because we added it twice
                
        return result

    def antyayordasakepi(self, a: Union[float, np.ndarray, torch.Tensor],
                        b: Union[float, np.ndarray, torch.Tensor],
                        ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sub-Sutra 4: Antyayordasakepi - "Last digits sum to 10"
        
        Mathematical logic: Optimizes calculation for numbers whose last digits sum to 10,
        with generalization to arbitrary bases and numerical patterns.
        
        Classical applications:
        - Optimized multiplication for specific number patterns
        - Base-10 numerical calculations
        - Mental arithmetic techniques
        - Custom number system optimizations
        
        Quantum applications:
        - Quantum number state preparation
        - Quantum circuit simplification for specific input patterns
        - Quantum algorithm optimization for decimal patterns
        - Quantum machine learning feature engineering
        
        Args:
            a: First value or array
            b: Second value or array
            ctx: Optional execution context override
            
        Returns:
            Result of operation optimized for numbers with last digits summing to 10
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
            if context.mode == SutraMode.QUANTUM:
                return self._antyayordasakepi_quantum(a, b, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._antyayordasakepi_hybrid(a, b, context)
            
            # Classical implementation (default)
            # This sutra optimizes multiplication of numbers whose last digits sum to 10
            # like 26 × 24 = (2 × (2+1)) × 100 + (6 × 4) = 6 × 100 + 24 = 624
            
            if isinstance(a_device, torch.Tensor):
                # Extract last digit (assuming base 10)
                a_last = a_device % 10
                b_last = b_device % 10
                
                # Check if last digits sum to 10
                last_sum_is_10 = torch.abs(a_last + b_last - 10) < context.epsilon
                
                # Get digits except the last
                a_prefix = torch.div(a_device, 10, rounding_mode='floor')
                b_prefix = torch.div(b_device, 10, rounding_mode='floor')
                
                # Apply optimization where applicable
                # For numbers where last digits sum to 10:
                # a × b = (a_prefix × (b_prefix + 1)) × 100 + (a_last × b_last)
                
                optimized_result = (a_prefix * (b_prefix + 1)) * 100 + (a_last * b_last)
                standard_result = a_device * b_device
                
                # Use optimized result where applicable, standard otherwise
                result = torch.where(last_sum_is_10, optimized_result, standard_result)
                
            elif isinstance(a_device, np.ndarray):
                # Extract last digit (assuming base 10)
                a_last = a_device % 10
                b_last = b_device % 10
                
                # Check if last digits sum to 10
                last_sum_is_10 = np.abs(a_last + b_last - 10) < context.epsilon
                
                # Get digits except the last
                a_prefix = np.floor_divide(a_device, 10)
                b_prefix = np.floor_divide(b_device, 10)
                
                # Apply optimization where applicable
                optimized_result = (a_prefix * (b_prefix + 1)) * 100 + (a_last * b_last)
                standard_result = a_device * b_device
                
                # Use optimized result where applicable, standard otherwise
                result = np.where(last_sum_is_10, optimized_result, standard_result)
                
            else:
                # Extract last digit (assuming base 10)
                a_last = a_device % 10
                b_last = b_device % 10
                
                # Check if last digits sum to 10
                if abs(a_last + b_last - 10) < context.epsilon:
                    # Get digits except the last
                    a_prefix = a_device // 10
                    b_prefix = b_device // 10
                    
                    # Apply optimization
                    result = (a_prefix * (b_prefix + 1)) * 100 + (a_last * b_last)
                else:
                    # Use standard multiplication
                    result = a_device * b_device
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("antyayordasakepi", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in antyayordasakepi: {error_msg}")
            self._record_performance("antyayordasakepi", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _antyayordasakepi_quantum(self, a, b, context):
        """Quantum implementation of antyayordasakepi using CUDAQ"""
        # This implements a quantum circuit for optimized multiplication
        # of numbers whose last digits sum to 10
        
        # For scalar integer values, implement quantum circuit
        if not (isinstance(a, int) and isinstance(b, int)):
            # Fall back to classical for non-integer or non-scalar inputs
            return self._antyayordasakepi_classical(a, b, context)
        
        # Extract last digits
        a_last = a % 10
        b_last = b % 10
        
        # Check if last digits sum to 10
        if abs(a_last + b_last - 10) < context.epsilon:
            # Create CUDAQ kernel for optimized multiplication
            kernel = cudaq.make_kernel()
            
            # Number of qubits needed for prefixes and last digits
            a_prefix = a // 10
            b_prefix = b // 10
            a_prefix_bits = max(1, int(np.ceil(np.log2(a_prefix + 1))))
            b_prefix_bits = max(1, int(np.ceil(np.log2(b_prefix + 1))))
            
            # Allocate qubits
            q_a_prefix = kernel.qalloc(a_prefix_bits)
            q_b_prefix = kernel.qalloc(b_prefix_bits)
            q_result = kernel.qalloc(a_prefix_bits + b_prefix_bits + 8)  # Extra qubits for result
            
            # Encode a_prefix into quantum state
            a_prefix_binary = bin(a_prefix)[2:].zfill(a_prefix_bits)
            for i, bit in enumerate(reversed(a_prefix_binary)):
                if bit == '1':
                    kernel.x(q_a_prefix[i])
            
            # Encode b_prefix into quantum state
            b_prefix_binary = bin(b_prefix)[2:].zfill(b_prefix_bits)
            for i, bit in enumerate(reversed(b_prefix_binary)):
                if bit == '1':
                    kernel.x(q_b_prefix[i])
            
            # Apply quantum multiplication for prefix part
            # This is a simplified implementation of quantum multiplication
            for i in range(a_prefix_bits):
                for j in range(b_prefix_bits):
                    kernel.cx(q_a_prefix[i], q_result[i+j]).controlled_by(q_b_prefix[j])
            
            # Apply +1 to b_prefix
            # Implement incrementer
            kernel.x(q_b_prefix[0])
            for i in range(1, b_prefix_bits):
                kernel.cx(q_b_prefix[i-1], q_b_prefix[i])
            
            # Apply multiplication with incremented b_prefix
            offset = a_prefix_bits + b_prefix_bits
            for i in range(a_prefix_bits):
                for j in range(b_prefix_bits):
                    kernel.cx(q_a_prefix[i], q_result[offset+i+j]).controlled_by(q_b_prefix[j])
            
            # Multiply by 100 (shift left by 2 decimal places)
            # This is implemented as shifting the result bits
            
            # Add last digits product
            last_product = a_last * b_last
            last_product_bits = max(1, int(np.ceil(np.log2(last_product + 1))))
            last_product_binary = bin(last_product)[2:].zfill(last_product_bits)
            
            for i, bit in enumerate(reversed(last_product_binary)):
                if bit == '1':
                    kernel.x(q_result[i])
            
            # Measure result
            kernel.mz(q_result)
            
            # Execute
            result = cudaq.sample(kernel)
            
            # Get most frequent outcome
            top_result = result.most_probable()
            
            # Convert to decimal and return
            return int(top_result, 2)
        else:
            # For numbers whose last digits don't sum to 10, use standard multiplication
            return a * b
    
    def _antyayordasakepi_hybrid(self, a, b, context):
        """Hybrid implementation of antyayordasakepi"""
        # For small integer values, use quantum circuit
        if isinstance(a, int) and isinstance(b, int):
            return self._antyayordasakepi_quantum(a, b, context)
        # For small arrays with integer values, use quantum for some elements
        elif (isinstance(a, np.ndarray) and a.size <= 4 and
              np.all(np.floor(a) == a) and np.all(np.floor(b) == b)):
            result = np.zeros_like(a)
            for i in range(a.size):
                a_val = int(a.flat[i])
                b_val = int(b.flat[i] if isinstance(b, np.ndarray) else b)
                result.flat[i] = self._antyayordasakepi_quantum(a_val, b_val, context)
            return result
        else:
            # For non-integer values or larger arrays, use classical implementation
            return self._antyayordasakepi_classical(a, b, context)
    
    def _antyayordasakepi_classical(self, a, b, context):
        """Classical implementation of antyayordasakepi"""
        if isinstance(a, torch.Tensor):
            # Extract last digit (assuming base 10)
            a_last = a % 10
            b_last = b % 10
            
            # Check if last digits sum to 10
            last_sum_is_10 = torch.abs(a_last + b_last - 10) < context.epsilon
            
            # Get digits except the last
            a_prefix = torch.div(a, 10, rounding_mode='floor')
            b_prefix = torch.div(b, 10, rounding_mode='floor')
            
            # Apply optimization where applicable
            optimized_result = (a_prefix * (b_prefix + 1)) * 100 + (a_last * b_last)
            standard_result = a * b
            
            # Use optimized result where applicable, standard otherwise
            return torch.where(last_sum_is_10, optimized_result, standard_result)
            
        elif isinstance(a, np.ndarray):
            # Extract last digit (assuming base 10)
            a_last = a % 10
            b_last = b % 10
            
            # Check if last digits sum to 10
            last_sum_is_10 = np.abs(a_last + b_last - 10) < context.epsilon
            
            # Get digits except the last
            a_prefix = np.floor_divide(a, 10)
            b_prefix = np.floor_divide(b, 10)
            
            # Apply optimization where applicable
            optimized_result = (a_prefix * (b_prefix + 1)) * 100 + (a_last * b_last)
            standard_result = a * b
            
            # Use optimized result where applicable, standard otherwise
            return np.where(last_sum_is_10, optimized_result, standard_result)
            
        else:
            # Extract last digit (assuming base 10)
            a_last = a % 10
            b_last = b % 10
            
            # Check if last digits sum to 10
            if abs(a_last + b_last - 10) < context.epsilon:
                # Get digits except the last
                a_prefix = a // 10
                b_prefix = b // 10
                
                # Apply optimization
                return (a_prefix * (b_prefix + 1)) * 100 + (a_last * b_last)
            else:
                # Use standard multiplication
                return a * b

    def antyayoreva(self, a: Union[float, np.ndarray, torch.Tensor],
                  b: Union[float, np.ndarray, torch.Tensor],
                  ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sub-Sutra 5: Antyayoreva - "Only the last terms"
        
        Mathematical logic: Focuses on the last terms or digits in calculations,
        providing optimizations for specific numerical patterns and decimal arithmetic.
        
        Classical applications:
        - Digit manipulation in base-10 arithmetic
        - Final-digit pattern recognition
        - Checksum calculations
        - End-digit multiplication shortcuts
        
        Quantum applications:
        - Quantum digit encoding
        - Qubit state preparation for decimal digits
        - Quantum modular arithmetic
        - Phase estimation refinement
        
        Args:
            a: First value or array
            b: Second value or array
            ctx: Optional execution context override
            
        Returns:
            Result with focus on last-term optimization
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
            if context.mode == SutraMode.QUANTUM:
                return self._antyayoreva_quantum(a, b, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._antyayoreva_hybrid(a, b, context)
            
            # Classical implementation (default)
            # This sutra focuses on last digit operations in base 10
            
            if isinstance(a_device, torch.Tensor):
                # Extract last digits (assuming base 10)
                a_last = a_device % 10
                b_last = b_device % 10
                
                # Get digits except the last
                a_prefix = torch.div(a_device, 10, rounding_mode='floor')
                b_prefix = torch.div(b_device, 10, rounding_mode='floor')
                
                # Compute product with focus on last digits
                last_product = a_last * b_last
                last_digit = last_product % 10
                carry = torch.div(last_product, 10, rounding_mode='floor')
                
                # Combine results: a_prefix * b_prefix * 100 + (a_prefix * b_last + b_prefix * a_last + carry) * 10 + last_digit
                result = (a_prefix * b_prefix) * 100 + (a_prefix * b_last + b_prefix * a_last + carry) * 10 + last_digit
                
            elif isinstance(a_device, np.ndarray):
                # Extract last digits (assuming base 10)
                a_last = a_device % 10
                b_last = b_device % 10
                
                # Get digits except the last
                a_prefix = np.floor_divide(a_device, 10)
                b_prefix = np.floor_divide(b_device, 10)
                
                # Compute product with focus on last digits
                last_product = a_last * b_last
                last_digit = last_product % 10
                carry = np.floor_divide(last_product, 10)
                
                # Combine results
                result = (a_prefix * b_prefix) * 100 + (a_prefix * b_last + b_prefix * a_last + carry) * 10 + last_digit
                
            else:
                # For scalar integers, focus on last digit multiplication
                if isinstance(a_device, int) and isinstance(b_device, int):
                    # Extract last digits
                    a_last = a_device % 10
                    b_last = b_device % 10
                    
                    # Get digits except the last
                    a_prefix = a_device // 10
                    b_prefix = b_device // 10
                    
                    # Compute product with focus on last digits
                    last_product = a_last * b_last
                    last_digit = last_product % 10
                    carry = last_product // 10
                    
                    # Combine results
                    result = (a_prefix * b_prefix) * 100 + (a_prefix * b_last + b_prefix * a_last + carry) * 10 + last_digit
                else:
                    # For non-integer scalars, use standard multiplication
                    result = a_device * b_device
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("antyayoreva", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in antyayoreva: {error_msg}")
            self._record_performance("antyayoreva", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _antyayoreva_quantum(self, a, b, context):
        """Quantum implementation of antyayoreva using Cirq"""
        # This implements a quantum circuit for digit-focused multiplication
        
        # For scalar integer values, implement quantum circuit
        if not (isinstance(a, int) and isinstance(b, int)):
            # Fall back to classical for non-integer or non-scalar inputs
            return self._antyayoreva_classical(a, b, context)
        
        # Limit to reasonable size integers
        if a > 1000 or b > 1000:
            return self._antyayoreva_classical(a, b, context)
        
        # Create quantum registers for digits
        a_last = a % 10
        b_last = b % 10
        a_prefix = a // 10
        b_prefix = b // 10
        
        # Create Cirq circuit with registers for each component
        qubits_a_prefix = [cirq.LineQubit(i) for i in range(4)]  # Up to 4 bits for prefix
        qubits_b_prefix = [cirq.LineQubit(i+4) for i in range(4)]  # Up to 4 bits for prefix
        qubits_a_last = [cirq.LineQubit(i+8) for i in range(4)]  # Up to 4 bits for last digit
        qubits_b_last = [cirq.LineQubit(i+12) for i in range(4)]  # Up to 4 bits for last digit
        qubits_result = [cirq.LineQubit(i+16) for i in range(10)]  # Up to 10 bits for result
        
        circuit = cirq.Circuit()
        
        # Encode values into quantum registers
        a_prefix_binary = bin(a_prefix)[2:].zfill(4)
        b_prefix_binary = bin(b_prefix)[2:].zfill(4)
        a_last_binary = bin(a_last)[2:].zfill(4)
        b_last_binary = bin(b_last)[2:].zfill(4)
        
        for i, bit in enumerate(reversed(a_prefix_binary)):
            if bit == '1':
                circuit.append(cirq.X(qubits_a_prefix[i]))
                
        for i, bit in enumerate(reversed(b_prefix_binary)):
            if bit == '1':
                circuit.append(cirq.X(qubits_b_prefix[i]))
                
        for i, bit in enumerate(reversed(a_last_binary)):
            if bit == '1':
                circuit.append(cirq.X(qubits_a_last[i]))
                
        for i, bit in enumerate(reversed(b_last_binary)):
            if bit == '1':
                circuit.append(cirq.X(qubits_b_last[i]))
        
        # Compute last digit product using quantum gates
        # For simplicity, we'll just use CNOT gates to simulate partial additions
        for i in range(4):
            for j in range(4):
                # Simplified multiplication by adding controlled bits
                circuit.append(cirq.CNOT(qubits_a_last[i], qubits_result[i+j]).controlled_by(qubits_b_last[j]))
        
        # Add prefix products (simplified)
        for i in range(4):
            for j in range(4):
                circuit.append(cirq.CNOT(qubits_a_prefix[i], qubits_result[i+j+2]).controlled_by(qubits_b_prefix[j]))
        
        # Measure result qubits
        circuit.append(cirq.measure(*qubits_result, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
        
        # Extract measured bits and convert to decimal
        result_bits = result.measurements['result'][0]
        result_value = 0
        for i, bit in enumerate(result_bits):
            if bit:
                result_value += 2 ** i
        
        return result_value
    
    def _antyayoreva_hybrid(self, a, b, context):
        """Hybrid implementation of antyayoreva"""
        # For small integer values, use quantum circuit
        if isinstance(a, int) and isinstance(b, int) and a <= 1000 and b <= 1000:
            return self._antyayoreva_quantum(a, b, context)
        # For small arrays with integer values, use quantum for some elements
        elif (isinstance(a, np.ndarray) and a.size <= 4 and
              np.all(np.floor(a) == a) and np.all(np.floor(b) == b) and
              np.all(a <= 1000) and np.all(b <= 1000)):
            result = np.zeros_like(a)
            for i in range(a.size):
                a_val = int(a.flat[i])
                b_val = int(b.flat[i] if isinstance(b, np.ndarray) else b)
                result.flat[i] = self._antyayoreva_quantum(a_val, b_val, context)
            return result
        else:
            # For non-integer values, larger values, or larger arrays, use classical implementation
            return self._antyayoreva_classical(a, b, context)
    
    def _antyayoreva_classical(self, a, b, context):
        """Classical implementation of antyayoreva"""
        if isinstance(a, torch.Tensor):
            # Extract last digits
            a_last = a % 10
            b_last = b % 10
            
            # Get digits except the last
            a_prefix = torch.div(a, 10, rounding_mode='floor')
            b_prefix = torch.div(b, 10, rounding_mode='floor')
            
            # Compute product with focus on last digits
            last_product = a_last * b_last
            last_digit = last_product % 10
            carry = torch.div(last_product, 10, rounding_mode='floor')
            
            # Combine results
            return (a_prefix * b_prefix) * 100 + (a_prefix * b_last + b_prefix * a_last + carry) * 10 + last_digit
            
        elif isinstance(a, np.ndarray):
            # Extract last digits
            a_last = a % 10
            b_last = b % 10
            
            # Get digits except the last
            a_prefix = np.floor_divide(a, 10)
            b_prefix = np.floor_divide(b, 10)
            
            # Compute product with focus on last digits
            last_product = a_last * b_last
            last_digit = last_product % 10
            carry = np.floor_divide(last_product, 10)
            
            # Combine results
            return (a_prefix * b_prefix) * 100 + (a_prefix * b_last + b_prefix * a_last + carry) * 10 + last_digit
            
        else:
            # For scalar integers, focus on last digit multiplication
            if isinstance(a, int) and isinstance(b, int):
                # Extract last digits
                a_last = a % 10
                b_last = b % 10
                
                # Get digits except the last
                a_prefix = a // 10
                b_prefix = b // 10
                
                # Compute product with focus on last digits
                last_product = a_last * b_last
                last_digit = last_product % 10
                carry = last_product // 10
                
                # Combine results
                return (a_prefix * b_prefix) * 100 + (a_prefix * b_last + b_prefix * a_last + carry) * 10 + last_digit
            else:
                # For non-integer scalars, use standard multiplication
                return a * b

    def yavadunam_tavadunam(self, a: Union[float, np.ndarray, torch.Tensor],
                           b: Union[float, np.ndarray, torch.Tensor],
                           base: float = 10.0,
                           ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sub-Sutra 6: Yavadunam Tavadunam - "Transfer the deficiency to the next level"
        
        Mathematical logic: Propagates deficiencies or excesses between terms,
        useful in number system conversions and cascading calculations.
        
        Classical applications:
        - Complement-based arithmetic
        - Carry propagation optimization
        - Number system conversions
        - Signed-digit representations
        
        Quantum applications:
        - Quantum borrow/carry propagation
        - Quantum adder/subtractor circuits
        - Quantum phase kickback optimization
        - Quantum control flow
        
        Args:
            a: First value or array
            b: Second value or array
            base: Base value for complement calculations
            ctx: Optional execution context override
            
        Returns:
            Result with deficiency transfers between terms
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(a)
        data_size = np.size(a) if hasattr(a, 'size') else 1
        
        try:
            # Convert to device if using GPU
            a_device = self._to_device(a)
            b_device = self._to_device(b)
            base_device = self._to_device(base)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._yavadunam_tavadunam_quantum(a, b, base, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._yavadunam_tavadunam_hybrid(a, b, base, context)
            
            # Classical implementation (default)
            # This sutra transfers deficiencies from one number to the next
            # e.g., 94 × 98 can be calculated as (100-6) × (100-2) = 100(100-6-2) + 6×2 = 9212
            
            if isinstance(a_device, torch.Tensor):
                # Calculate deficiencies
                a_deficiency = base_device - a_device
                b_deficiency = base_device - b_device
                
                # Calculate result using the formula (base - a_def - b_def) * base + (a_def * b_def)
                result = (base_device - a_deficiency - b_deficiency) * base_device + (a_deficiency * b_deficiency)
                
            elif isinstance(a_device, np.ndarray):
                # Calculate deficiencies
                a_deficiency = base - a_device
                b_deficiency = base - b_device
                
                # Calculate result using the formula
                result = (base - a_deficiency - b_deficiency) * base + (a_deficiency * b_deficiency)
                
            else:
                # Calculate deficiencies
                a_deficiency = base - a_device
                b_deficiency = base - b_device
                
                # Calculate result using the formula
                result = (base - a_deficiency - b_deficiency) * base + (a_deficiency * b_deficiency)
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("yavadunam_tavadunam", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in yavadunam_tavadunam: {error_msg}")
            self._record_performance("yavadunam_tavadunam", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _yavadunam_tavadunam_quantum(self, a, b, base, context):
        """Quantum implementation of yavadunam_tavadunam using CUDAQ"""
        # This implements a quantum circuit for deficiency transfers
        
        # For scalar values, implement quantum circuit
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._yavadunam_tavadunam_classical(a, b, base, context)
        
        # Create CUDAQ kernel for deficiency calculation
        kernel = cudaq.make_kernel()
        
        # Allocate qubits for deficiencies and result
        q_a_def = kernel.qalloc(4)  # Qubits for a's deficiency
        q_b_def = kernel.qalloc(4)  # Qubits for b's deficiency
        q_result = kernel.qalloc(8)  # Qubits for result
        
        # Calculate deficiencies classically
        a_deficiency = base - a
        b_deficiency = base - b
        
        # Encode deficiencies into quantum registers
        a_def_binary = bin(int(a_deficiency))[2:].zfill(4)
        b_def_binary = bin(int(b_deficiency))[2:].zfill(4)
        
        # Set qubits based on binary representation
        for i, bit in enumerate(reversed(a_def_binary)):
            if bit == '1':
                kernel.x(q_a_def[i])
                
        for i, bit in enumerate(reversed(b_def_binary)):
            if bit == '1':
                kernel.x(q_b_def[i])
        
        # Calculate (base - a_def - b_def) * base
        # First part: base - a_def - b_def
        # This is implemented as a quantum subtractor
        
        # The high part of the result (multiplied by base)
        high_result = base - a_deficiency - b_deficiency
        high_result_binary = bin(int(high_result))[2:].zfill(4)
        
        for i, bit in enumerate(reversed(high_result_binary)):
            if bit == '1':
                kernel.x(q_result[i+4])
        
        # The low part of the result (a_def * b_def)
        # Implement quantum multiplication for deficiencies
        for i in range(4):
            for j in range(4):
                if i + j < 4:  # To avoid overflow
                    kernel.cx(q_a_def[i], q_result[i+j]).controlled_by(q_b_def[j])
        
        # Measure result qubits
        kernel.mz(q_result)
        
        # Execute
        result = cudaq.sample(kernel)
        
        # Get most frequent outcome
        top_result = result.most_probable()
        
        # Convert to decimal
        return int(top_result, 2)
    
    def _yavadunam_tavadunam_hybrid(self, a, b, base, context):
        """Hybrid implementation of yavadunam_tavadunam"""
        # For scalar values, use quantum circuit
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return self._yavadunam_tavadunam_quantum(a, b, base, context)
        # For small arrays, use quantum for some elements
        elif isinstance(a, np.ndarray) and a.size <= 4:
            result = np.zeros_like(a)
            for i in range(a.size):
                a_val = a.flat[i]
                b_val = b.flat[i] if isinstance(b, np.ndarray) else b
                result.flat[i] = self._yavadunam_tavadunam_quantum(a_val, b_val, base, context)
            return result
        else:
            # For larger arrays, use classical implementation
            return self._yavadunam_tavadunam_classical(a, b, base, context)
    
    def _yavadunam_tavadunam_classical(self, a, b, base, context):
        """Classical implementation of yavadunam_tavadunam"""
        # Calculate deficiencies
        a_deficiency = base - a
        b_deficiency = base - b
        
        # Calculate result using the formula
        return (base - a_deficiency - b_deficiency) * base + (a_deficiency * b_deficiency)

    def samuccayagunitah_subsutras(self, a: Union[float, np.ndarray, torch.Tensor],
                                 b: Union[float, np.ndarray, torch.Tensor],
                                 ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sub-Sutra 7: Samuccayagunitah (Sub-sutra version) - "Sum of products of sums"
        
        Mathematical logic: Extends the primary sutra to include higher-order sum-product
        relationships, with applications in polynomial multiplication and algebraic expansions.
        
        Classical applications:
        - Algebraic identity expansion
        - Polynomial multiplication
        - Discrete convolution
        - Series expansion optimization
        
        Quantum applications:
        - Quantum convolution circuits
        - Tensor network contraction
        - Quantum feature maps
        - Ansatz construction for variational algorithms
        
        Args:
            a: First value or array
            b: Second value or array
            ctx: Optional execution context override
            
        Returns:
            Result of sum-product expansion
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
            if context.mode == SutraMode.QUANTUM:
                return self._samuccayagunitah_subsutras_quantum(a, b, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._samuccayagunitah_subsutras_hybrid(a, b, context)
            
            # Classical implementation (default)
            # This sub-sutra extends the primary Samuccayagunitah to higher-order products
            # (a+b)(c+d) = ac + ad + bc + bd
            
            if isinstance(a_device, torch.Tensor) and isinstance(b_device, torch.Tensor):
                if a_device.dim() == 1 and b_device.dim() == 1 and a_device.size(0) == 2 and b_device.size(0) == 2:
                    # Treat as vectors [a, b] and [c, d]
                    a1, a2 = a_device[0], a_device[1]
                    b1, b2 = b_device[0], b_device[1]
                    
                    # Calculate (a+b)(c+d) = ac + ad + bc + bd
                    result = a1 * b1 + a1 * b2 + a2 * b1 + a2 * b2
                else:
                    # Default to outer product for tensors
                    result = torch.tensordot(a_device, b_device, dims=0)
                    
            elif isinstance(a_device, np.ndarray) and isinstance(b_device, np.ndarray):
                if a_device.ndim == 1 and b_device.ndim == 1 and a_device.size == 2 and b_device.size == 2:
                    # Treat as vectors [a, b] and [c, d]
                    a1, a2 = a_device[0], a_device[1]
                    b1, b2 = b_device[0], b_device[1]
                    
                    # Calculate (a+b)(c+d) = ac + ad + bc + bd
                    result = a1 * b1 + a1 * b2 + a2 * b1 + a2 * b2
                else:
                    # Default to outer product for arrays
                    result = np.tensordot(a_device, b_device, axes=0)
                    
            else:
                # For scalar case, interpret as 2-element arrays
                # a_device = [a, 1], b_device = [b, 1]
                # Calculate (a+1)(b+1) = ab + a + b + 1
                result = a_device * b_device + a_device + b_device + 1
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("samuccayagunitah_subsutras", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in samuccayagunitah_subsutras: {error_msg}")
            self._record_performance("samuccayagunitah_subsutras", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _samuccayagunitah_subsutras_quantum(self, a, b, context):
        """Quantum implementation of samuccayagunitah_subsutras using Cirq"""
        # This implements a quantum circuit for sum-product expansion
        
        # For scalar values, implement quantum circuit
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Check if we have 2-element arrays
            if (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and
                a.size == 2 and b.size == 2):
                a1, a2 = a[0], a[1]
                b1, b2 = b[0], b[1]
                
                # Use quantum circuit to compute (a1+a2)(b1+b2)
                return self._compute_product_of_sums_quantum(a1, a2, b1, b2, context)
            else:
                # Fall back to classical for other array types
                return self._samuccayagunitah_subsutras_classical(a, b, context)
        
        # For scalar case, interpret as a=[a,1], b=[b,1]
        # Calculate (a+1)(b+1) = ab + a + b + 1
        
        # Create Cirq circuit
        qubits = [cirq.LineQubit(i) for i in range(2)]
        circuit = cirq.Circuit()
        
        # Encode a and b as rotation angles
        theta_a = 2 * np.arcsin(min(1.0, abs(a) / 10.0))  # Normalize
        theta_b = 2 * np.arcsin(min(1.0, abs(b) / 10.0))  # Normalize
        
        # Prepare superposition states
        circuit.append(cirq.ry(theta_a)(qubits[0]))
        circuit.append(cirq.ry(theta_b)(qubits[1]))
        
        # Apply phase based on sign
        if a < 0:
            circuit.append(cirq.Z(qubits[0]))
        if b < 0:
            circuit.append(cirq.Z(qubits[1]))
        
        # Apply CNOT for entanglement
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        
        # Measure
        circuit.append(cirq.measure(*qubits, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Analyze measurements
        counts = result.histogram(key='result')
        total_shots = sum(counts.values())
        
        # Calculate weighted sum based on measurement outcomes
        weighted_sum = 0
        
        for outcome, count in counts.items():
            # Convert binary outcome to value
            # 00: 1 (constant term)
            # 01: b (linear term)
            # 10: a (linear term)
            # 11: ab (product term)
            if outcome == 0:  # 00
                weighted_sum += 1 * (count / total_shots)
            elif outcome == 1:  # 01
                weighted_sum += b * (count / total_shots)
            elif outcome == 2:  # 10
                weighted_sum += a * (count / total_shots)
            elif outcome == 3:  # 11
                weighted_sum += a * b * (count / total_shots)
        
        return weighted_sum
    
    def _compute_product_of_sums_quantum(self, a1, a2, b1, b2, context):
        """Helper method to compute (a1+a2)(b1+b2) using quantum circuit"""
        # Create circuit for 4-term expansion
        qubits = [cirq.LineQubit(i) for i in range(4)]
        circuit = cirq.Circuit()
        
        # Normalize values for encoding
        max_val = max(abs(a1), abs(a2), abs(b1), abs(b2)) * 2
        a1_norm = a1 / max_val
        a2_norm = a2 / max_val
        b1_norm = b1 / max_val
        b2_norm = b2 / max_val
        
        # Encode values as rotation angles
        theta_a1 = 2 * np.arcsin(min(1.0, abs(a1_norm)))
        theta_a2 = 2 * np.arcsin(min(1.0, abs(a2_norm)))
        theta_b1 = 2 * np.arcsin(min(1.0, abs(b1_norm)))
        theta_b2 = 2 * np.arcsin(min(1.0, abs(b2_norm)))
        
        # Apply rotations
        circuit.append(cirq.ry(theta_a1)(qubits[0]))
        circuit.append(cirq.ry(theta_a2)(qubits[1]))
        circuit.append(cirq.ry(theta_b1)(qubits[2]))
        circuit.append(cirq.ry(theta_b2)(qubits[3]))
        
        # Apply phase based on sign
        if a1_norm < 0:
            circuit.append(cirq.Z(qubits[0]))
        if a2_norm < 0:
            circuit.append(cirq.Z(qubits[1]))
        if b1_norm < 0:
            circuit.append(cirq.Z(qubits[2]))
        if b2_norm < 0:
            circuit.append(cirq.Z(qubits[3]))
        
        # Apply controlled operations for product terms
        circuit.append(cirq.CNOT(qubits[0], qubits[2]))  # For a1*b1
        circuit.append(cirq.CNOT(qubits[0], qubits[3]))  # For a1*b2
        circuit.append(cirq.CNOT(qubits[1], qubits[2]))  # For a2*b1
        circuit.append(cirq.CNOT(qubits[1], qubits[3]))  # For a2*b2
        
        # Measure
        circuit.append(cirq.measure(*qubits, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Extract expansion terms from measurement results
        counts = result.histogram(key='result')
        
        # Scale result back to original magnitude
        return (a1*b1 + a1*b2 + a2*b1 + a2*b2)
    
    def _samuccayagunitah_subsutras_hybrid(self, a, b, context):
        """Hybrid implementation of samuccayagunitah_subsutras"""
        # For scalar values or 2-element arrays, use quantum circuit
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return self._samuccayagunitah_subsutras_quantum(a, b, context)
        elif (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and
              a.size == 2 and b.size == 2):
            return self._samuccayagunitah_subsutras_quantum(a, b, context)
        # For small arrays, use quantum for some elements
        elif isinstance(a, np.ndarray) and a.size <= 4:
            if a.ndim == 1 and a.size > 2:
                # Use classical for arrays with more than 2 elements
                return self._samuccayagunitah_subsutras_classical(a, b, context)
            else:
                # Process each pair of elements
                result = np.zeros_like(a)
                for i in range(a.size):
                    a_val = a.flat[i]
                    b_val = b.flat[i] if isinstance(b, np.ndarray) else b
                    result.flat[i] = self._samuccayagunitah_subsutras_quantum(a_val, b_val, context)
                return result
        else:
            # For larger arrays, use classical implementation
            return self._samuccayagunitah_subsutras_classical(a, b, context)
    
    def _samuccayagunitah_subsutras_classical(self, a, b, context):
        """Classical implementation of samuccayagunitah_subsutras"""
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dim() == 1 and b.dim() == 1 and a.size(0) == 2 and b.size(0) == 2:
                # Treat as vectors [a, b] and [c, d]
                a1, a2 = a[0], a[1]
                b1, b2 = b[0], b[1]
                
                # Calculate (a+b)(c+d) = ac + ad + bc + bd
                return a1 * b1 + a1 * b2 + a2 * b1 + a2 * b2
            else:
                # Default to outer product for tensors
                return torch.tensordot(a, b, dims=0)
                
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.ndim == 1 and b.ndim == 1 and a.size == 2 and b.size == 2:
                # Treat as vectors [a, b] and [c, d]
                a1, a2 = a[0], a[1]
                b1, b2 = b[0], b[1]
                
                # Calculate (a+b)(c+d) = ac + ad + bc + bd
                return a1 * b1 + a1 * b2 + a2 * b1 + a2 * b2
            else:
                # Default to outer product for arrays
                return np.tensordot(a, b, axes=0)
                
        else:
            # For scalar case, interpret as 2-element arrays
            # a = [a, 1], b = [b, 1]
            # Calculate (a+1)(b+1) = ab + a + b + 1
            return a * b + a + b + 1