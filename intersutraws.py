# ========== INTER-SUTRA INTERACTIONS ==========
    
    def apply_sutra_sequence(self, x: Union[float, np.ndarray, torch.Tensor],
                             sutra_sequence: List[Tuple[str, Dict[str, Any]]],
                             ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Applies a sequence of sutras to the input data, enabling complex
        transformations through sutra composition.
        
        This method is the core of the multi-sutra synergy engine, allowing
        different sutras to be applied in specific sequences to achieve
        advanced computational effects.
        
        Args:
            x: Input value or array
            sutra_sequence: List of (sutra_name, params_dict) tuples defining the sequence
            ctx: Optional execution context override
            
        Returns:
            Result after applying the sutra sequence
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            result = x_device
            
            # Track sutras used and their performance
            sutras_used = []
            
            # Apply each sutra in sequence
            for sutra_name, params in sutra_sequence:
                # Get the method corresponding to the sutra name
                sutra_method = getattr(self, sutra_name, None)
                
                if sutra_method is None:
                    raise ValueError(f"Unknown sutra: {sutra_name}")
                
                # Apply the sutra with the given parameters
                sutra_start_time = time.time()
                result = sutra_method(result, **params, ctx=context)
                sutra_end_time = time.time()
                
                # Record usage
                sutras_used.append({
                    'sutra': sutra_name,
                    'execution_time': sutra_end_time - sutra_start_time,
                    'params': params
                })
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            # Record interaction
            self.sutra_interactions[tuple(s['sutra'] for s in sutras_used)] = {
                'count': self.sutra_interactions.get(tuple(s['sutra'] for s in sutras_used), {}).get('count', 0) + 1,
                'avg_execution_time': sum(s['execution_time'] for s in sutras_used) / len(sutras_used)
            }
            
            end_time = time.time()
            self._record_performance("apply_sutra_sequence", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in apply_sutra_sequence: {error_msg}")
            self._record_performance("apply_sutra_sequence", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def recommend_sutra_sequence(self, problem_type: str, 
                                data_shape: Optional[Tuple[int, ...]] = None,
                                data_characteristics: Optional[Dict[str, Any]] = None,
                                ctx: Optional[SutraContext] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Recommends an optimal sequence of sutras for a given problem type
        and data characteristics.
        
        This intelligent selector leverages known sutra synergies and performance
        history to suggest the most effective sequence of sutras for a specific
        computational task.
        
        Args:
            problem_type: Type of problem (e.g., 'PDE', 'NP-hard', 'quantum_optimization')
            data_shape: Shape of the data (if applicable)
            data_characteristics: Additional characteristics of the data
            ctx: Optional execution context override
            
        Returns:
            Recommended sequence of sutras as a list of (sutra_name, params_dict) tuples
        """
        context = ctx or self.context
        start_time = time.time()
        
        try:
            # Define known effective sutra sequences for different problem types
            effective_sequences = {
                'PDE': [
                    ('urdhva_tiryagbhyam', {'a': None, 'b': None}),
                    ('paravartya_yojayet', {'x': None, 'divisor': 2.0}),
                    ('maya_illusion_transform', {'phase_factor': 0.3, 'frequency': 2.0})
                ],
                'NP-hard': [
                    ('nikhilam_navatashcaramam_dashatah', {'x': None, 'base': 10.0}),
                    ('ekadhikena_purvena', {'x': None, 'iterations': 1}),
                    ('anurupyena', {'a': None, 'b': 1.0, 'ratio': 0.618})
                ],
                'quantum_optimization': [
                    ('shunyam_samyasamuccaye', {'a': None, 'b': None}),
                    ('vyashtisamanstih', {'whole': None, 'parts': None}),
                    ('paravartya_yojayet', {'x': None, 'divisor': 1.0})
                ],
                'matrix_operations': [
                    ('urdhva_tiryagbhyam', {'a': None, 'b': None}),
                    ('sesanyankena_caramena', {'coefficients': None, 'x': None}),
                    ('samuccayagunitah', {'a': None, 'b': None, 'operation': 'product_sum'})
                ],
                'signal_processing': [
                    ('maya_illusion_transform', {'phase_factor': 0.5, 'frequency': 1.0}),
                    ('maya_illusion_phase_cancellation', {'phase_factor': 0.3, 'frequency': 2.0, 'threshold': 0.1}),
                    ('chalana_kalana', {'x': None, 'steps': 3, 'direction': 1})
                ],
                'geometric_calculations': [
                    ('sulba_geometric_mean', {'a': None, 'b': None}),
                    ('sulba_square_construction', {'side_length': None}),
                    ('sulba_circle_construction', {'radius': None})
                ],
                'default': [
                    ('nikhilam_navatashcaramam_dashatah', {'x': None, 'base': 10.0}),
                    ('urdhva_tiryagbhyam', {'a': None, 'b': None})
                ]
            }
            
            # Get base sequence for the problem type
            base_sequence = effective_sequences.get(problem_type, effective_sequences['default'])
            
            # Customize sequence based on data characteristics if provided
            if data_characteristics:
                # Example customization based on data characteristics
                if data_characteristics.get('sparsity', 0) < 0.3:
                    # For sparse data, add sutras good for sparse computations
                    base_sequence.append(('anurupye_sunyamanyat', {'a': None, 'b': None, 'threshold': 0.01}))
                
                if data_characteristics.get('dimensionality', 0) > 2:
                    # For high-dimensional data, add specific sutras
                    base_sequence.append(('vyashtisamanstih', {'whole': None, 'parts': None}))
                
                if data_characteristics.get('periodicity', 0) > 0.5:
                    # For periodic data, add phase-based sutras
                    base_sequence.append(('maya_illusion_multi_layer', {
                        'phase_factors': [0.3, 0.5, 0.7],
                        'frequencies': [1.0, 2.0, 3.0]
                    }))
            
            # Adapt to data shape if provided
            if data_shape:
                if len(data_shape) == 1:  # Vector
                    if data_shape[0] > 1000:
                        # Large vectors benefit from Ekadhikena for progressive processing
                        base_sequence.insert(0, ('ekadhikena_purvena', {'x': None, 'iterations': 1}))
                elif len(data_shape) == 2:  # Matrix
                    if data_shape[0] > 100 and data_shape[1] > 100:
                        # Large matrices benefit from Urdhva for efficient multiplication
                        if not any(s[0] == 'urdhva_tiryagbhyam' for s in base_sequence):
                            base_sequence.insert(0, ('urdhva_tiryagbhyam', {'a': None, 'b': None}))
                elif len(data_shape) > 2:  # Tensor
                    # High-dimensional tensors benefit from specialized sutras
                    base_sequence.append(('gunakasamuccayah', {'a': None, 'b': None}))
            
            # Adapt to context-specific optimizations
            if context.mode == SutraMode.QUANTUM:
                # For quantum mode, prioritize quantum-friendly sutras
                quantum_sutras = [
                    ('shunyam_samyasamuccaye', {'a': None, 'b': None}),
                    ('paravartya_yojayet', {'x': None, 'divisor': 2.0})
                ]
                base_sequence = quantum_sutras + base_sequence
            elif context.mode == SutraMode.MAYA_ILLUSION:
                # For maya illusion mode, prioritize illusion sutras
                illusion_sutras = [
                    ('maya_illusion_transform', {'phase_factor': 0.5, 'frequency': 1.0}),
                    ('maya_illusion_multi_layer', {
                        'phase_factors': [0.3, 0.5, 0.7],
                        'frequencies': [1.0, 2.0, 3.0]
                    })
                ]
                base_sequence = illusion_sutras + base_sequence
            
            # Use performance history to optimize sequence if available
            if self.performance_history:
                # Group performance by sutra
                sutra_performance = {}
                for record in self.performance_history:
                    if record['success']:
                        sutra = record['sutra']
                        if sutra not in sutra_performance:
                            sutra_performance[sutra] = []
                        sutra_performance[sutra].append(record['execution_time'])
                
                # Calculate average execution time
                avg_execution_time = {s: sum(times) / len(times) for s, times in sutra_performance.items()}
                
                # Sort sutras by performance
                sorted_sutras = sorted(avg_execution_time.keys(), key=lambda s: avg_execution_time[s])
                
                # Prioritize faster sutras
                for fast_sutra in sorted_sutras[:3]:  # Top 3 fastest
                    if not any(s[0] == fast_sutra for s in base_sequence):
                        # Add fast sutra with default parameters
                        base_sequence.insert(0, (fast_sutra, {'x': None}))
            
            end_time = time.time()
            self._record_performance("recommend_sutra_sequence", start_time, end_time, 
                                    True, 1)
            return base_sequence
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in recommend_sutra_sequence: {error_msg}")
            self._record_performance("recommend_sutra_sequence", start_time, end_time, 
                                   False, 1)
            raise
    
    def optimize_sutra_sequence(self, initial_sequence: List[Tuple[str, Dict[str, Any]]],
                               test_data: Union[float, np.ndarray, torch.Tensor],
                               target_output: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
                               iterations: int = 10,
                               ctx: Optional[SutraContext] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Optimizes a sutra sequence by iteratively testing and refining the sequence
        to improve performance or accuracy.
        
        This method implements a search process to find the optimal sequence and
        parameters for sutras to solve a specific problem instance.
        
        Args:
            initial_sequence: Starting sutra sequence
            test_data: Data to test the sequence on
            target_output: Expected output (if available, for supervised optimization)
            iterations: Number of optimization iterations
            ctx: Optional execution context override
            
        Returns:
            Optimized sutra sequence
        """
        context = ctx or self.context
        start_time = time.time()
        
        try:
            # Initialize with the provided sequence
            best_sequence = initial_sequence.copy()
            best_result = None
            best_error = float('inf')
            
            # If target output is provided, we can optimize for accuracy
            if target_output is not None:
                # Convert target to device if using GPU
                target_device = self._to_device(target_output)
                
                # Apply initial sequence to get baseline
                baseline = test_data
                for sutra_name, params in best_sequence:
                    sutra_method = getattr(self, sutra_name, None)
                    if sutra_method is not None:
                        baseline = sutra_method(baseline, **params, ctx=context)
                
                # Calculate initial error
                if isinstance(baseline, torch.Tensor) and isinstance(target_device, torch.Tensor):
                    best_error = torch.mean(torch.abs(baseline - target_device)).item()
                elif isinstance(baseline, np.ndarray) and isinstance(target_device, np.ndarray):
                    best_error = np.mean(np.abs(baseline - target_device))
                else:
                    best_error = abs(baseline - target_device)
                
                best_result = baseline
            else:
                # Without target output, optimize for execution time
                # Apply initial sequence to get baseline timing
                timing_start = time.time()
                baseline = test_data
                for sutra_name, params in best_sequence:
                    sutra_method = getattr(self, sutra_name, None)
                    if sutra_method is not None:
                        baseline = sutra_method(baseline, **params, ctx=context)
                timing_end = time.time()
                
                best_error = timing_end - timing_start  # "Error" is execution time
                best_result = baseline
            
            # List of available sutras for exploration
            available_sutras = [
                # Primary sutras
                'ekadhikena_purvena',
                'nikhilam_navatashcaramam_dashatah',
                'urdhva_tiryagbhyam',
                'paravartya_yojayet',
                'shunyam_samyasamuccaye',
                'vyashtisamanstih',
                'chalana_kalana',
                'sankalana_vyavakalanabhyam',
                'purna_apurna_bhyam',
                'sesanyankena_caramena',
                'ekanyunena_purvena',
                'anurupyena',
                'gunakasamuccayah',
                'yavadunam',
                'samuccayagunitah',
                'gunakasamuccayah',
                
                # Sub sutras
                'anurupye_sunyamanyat',
                'sisyate_sesasamjnah',
                'adyamadyenantyamantyena',
                'antyayordasakepi',
                'antyayoreva',
                'yavadunam_tavadunam',
                'samuccayagunitah_subsutras',
                
                # Maya and Sulba sutras
                'maya_illusion_transform',
                'maya_illusion_multi_layer',
                'maya_illusion_phase_cancellation',
                'sulba_square_construction',
                'sulba_circle_construction',
                'sulba_geometric_mean'
            ]
            
            # Iterate to optimize
            for iteration in range(iterations):
                # Try different optimization strategies in each iteration
                strategy = iteration % 3
                
                if strategy == 0:
                    # Strategy 1: Modify sequence order
                    if len(best_sequence) > 1:
                        # Try swapping two sutras
                        for i in range(len(best_sequence)):
                            for j in range(i+1, len(best_sequence)):
                                # Create new sequence with swapped sutras
                                new_sequence = best_sequence.copy()
                                new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
                                
                                # Evaluate new sequence
                                if self._evaluate_sequence(new_sequence, test_data, target_device, context) < best_error:
                                    best_sequence = new_sequence
                                    best_error = self._evaluate_sequence(new_sequence, test_data, target_device, context)
                
                elif strategy == 1:
                    # Strategy 2: Add or remove a sutra
                    # Try adding a new sutra
                    for sutra in available_sutras:
                        if not any(s[0] == sutra for s in best_sequence):
                            # Add sutra with default parameters
                            new_sequence = best_sequence + [(sutra, {'x': None})]
                            
                            # Evaluate new sequence
                            if self._evaluate_sequence(new_sequence, test_data, target_device, context) < best_error:
                                best_sequence = new_sequence
                                best_error = self._evaluate_sequence(new_sequence, test_data, target_device, context)
                    
                    # Try removing a sutra
                    if len(best_sequence) > 1:
                        for i in range(len(best_sequence)):
                            new_sequence = best_sequence.copy()
                            del new_sequence[i]
                            
                            # Evaluate new sequence
                            if self._evaluate_sequence(new_sequence, test_data, target_device, context) < best_error:
                                best_sequence = new_sequence
                                best_error = self._evaluate_sequence(new_sequence, test_data, target_device, context)
                
                else:
                    # Strategy 3: Modify parameters
                    for i, (sutra_name, params) in enumerate(best_sequence):
                        # Create variations of parameters
                        new_params = params.copy()
                        
                        # Modify numeric parameters
                        for param_name, value in params.items():
                            if isinstance(value, (int, float)) and param_name != 'x':
                                # Try increasing and decreasing by 10%
                                for factor in [0.9, 1.1]:
                                    new_params[param_name] = value * factor
                                    new_sequence = best_sequence.copy()
                                    new_sequence[i] = (sutra_name, new_params.copy())
                                    
                                    # Evaluate new sequence
                                    if self._evaluate_sequence(new_sequence, test_data, target_device, context) < best_error:
                                        best_sequence = new_sequence
                                        best_error = self._evaluate_sequence(new_sequence, test_data, target_device, context)
            
            end_time = time.time()
            self._record_performance("optimize_sutra_sequence", start_time, end_time, 
                                    True, 1)
            return best_sequence
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in optimize_sutra_sequence: {error_msg}")
            self._record_performance("optimize_sutra_sequence", start_time, end_time, 
                                   False, 1)
            raise
    
    def _evaluate_sequence(self, sequence, test_data, target_output, context):
        """Helper method to evaluate a sutra sequence"""
        try:
            # Apply sequence
            timing_start = time.time()
            result = test_data
            for sutra_name, params in sequence:
                # Get the method corresponding to the sutra name
                sutra_method = getattr(self, sutra_name, None)
                if sutra_method is not None:
                    # Prepare parameters (replace None with appropriate value)
                    actual_params = params.copy()
                    if 'x' in actual_params and actual_params['x'] is None:
                        actual_params['x'] = result
                    elif 'a' in actual_params and actual_params['a'] is None:
                        actual_params['a'] = result
                    
                    # Apply the sutra
                    result = sutra_method(**actual_params, ctx=context)
            timing_end = time.time()
            
            # Calculate error or timing
            if target_output is not None:
                # Calculate error against target
                if isinstance(result, torch.Tensor) and isinstance(target_output, torch.Tensor):
                    error = torch.mean(torch.abs(result - target_output)).item()
                elif isinstance(result, np.ndarray) and isinstance(target_output, np.ndarray):
                    error = np.mean(np.abs(result - target_output))
                else:
                    error = abs(result - target_output)
            else:
                # Use timing as "error"
                error = timing_end - timing_start
            
            return error
            
        except Exception as e:
            # If execution fails, return infinity to discourage this sequence
            logger.warning(f"Sequence evaluation failed: {e}")
            return float('inf')