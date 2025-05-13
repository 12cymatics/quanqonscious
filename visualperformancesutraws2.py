def visualize_performance(self, n_top: int = 10, output_format: str = 'matplotlib'):
        """
        Visualizes the performance of sutras based on execution history.
        
        This method generates visualizations to help analyze the performance
        and effectiveness of different sutras and their combinations.
        
        Args:
            n_top: Number of top sutras to include in visualizations
            output_format: Format for visualization ('matplotlib' or 'text')
            
        Returns:
            Visualization data or plots
        """
        try:
            if not self.performance_history:
                return "No performance data available."
            
            # Group performance data by sutra
            sutra_performance = {}
            for record in self.performance_history:
                sutra = record['sutra']
                success = record['success']
                time = record['execution_time']
                
                if sutra not in sutra_performance:
                    sutra_performance[sutra] = {
                        'times': [],
                        'success_count': 0,
                        'failure_count': 0
                    }
                
                sutra_performance[sutra]['times'].append(time)
                if success:
                    sutra_performance[sutra]['success_count'] += 1
                else:
                    sutra_performance[sutra]['failure_count'] += 1
            
            # Calculate average execution times
            avg_times = {}
            success_rates = {}
            
            for sutra, data in sutra_performance.items():
                total_count = data['success_count'] + data['failure_count']
                if total_count > 0:
                    success_rates[sutra] = data['success_count'] / total_count * 100
                    
                if data['times']:
                    avg_times[sutra] = sum(data['times']) / len(data['times'])
            
            # Get top sutras by execution time
            top_sutras_by_time = sorted(avg_times.keys(), key=lambda s: avg_times[s])[:n_top]
            
            # Get top sutras by success rate
            top_sutras_by_success = sorted(success_rates.keys(), key=lambda s: success_rates[s], reverse=True)[:n_top]
            
            # Analyze sutra interactions
            interaction_performance = {}
            for interaction, data in self.sutra_interactions.items():
                interaction_performance[interaction] = {
                    'count': data['count'],
                    'avg_time': data['avg_execution_time']
                }
            
            # Get top interactions by frequency
            top_interactions_by_freq = sorted(interaction_performance.keys(), key=lambda i: interaction_performance[i]['count'], reverse=True)[:n_top]
            
            # Get top interactions by execution time
            top_interactions_by_time = sorted(interaction_performance.keys(), key=lambda i: interaction_performance[i]['avg_time'])[:n_top]
            
            # Generate visualizations
            if output_format == 'matplotlib':
                import matplotlib.pyplot as plt
                
                # Create figure with multiple subplots
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Plot 1: Average execution times
                axes[0, 0].barh([s for s in top_sutras_by_time], [avg_times[s] for s in top_sutras_by_time])
                axes[0, 0].set_title('Top Sutras by Execution Time')
                axes[0, 0].set_xlabel('Average Execution Time (s)')
                axes[0, 0].set_ylabel('Sutra')
                
                # Plot 2: Success rates
                axes[0, 1].barh([s for s in top_sutras_by_success], [success_rates[s] for s in top_sutras_by_success])
                axes[0, 1].set_title('Top Sutras by Success Rate')
                axes[0, 1].set_xlabel('Success Rate (%)')
                axes[0, 1].set_ylabel('Sutra')
                
                # Plot 3: Interaction frequencies
                if top_interactions_by_freq:
                    axes[1, 0].barh([' + '.join(i) for i in top_interactions_by_freq], 
                                   [interaction_performance[i]['count'] for i in top_interactions_by_freq])
                    axes[1, 0].set_title('Top Sutra Interactions by Frequency')
                    axes[1, 0].set_xlabel('Interaction Count')
                    axes[1, 0].set_ylabel('Sutra Sequence')
                
                # Plot 4: Interaction execution times
                if top_interactions_by_time:
                    axes[1, 1].barh([' + '.join(i) for i in top_interactions_by_time], 
                                   [interaction_performance[i]['avg_time'] for i in top_interactions_by_time])
                    axes[1, 1].set_title('Top Sutra Interactions by Execution Time')
                    axes[1, 1].set_xlabel('Average Execution Time (s)')
                    axes[1, 1].set_ylabel('Sutra Sequence')
                
                plt.tight_layout()
                return fig
                
            else:  # Text output
                output = []
                
                output.append("===== SUTRA PERFORMANCE ANALYSIS =====\n")
                
                output.append("Top Sutras by Execution Time:")
                for i, sutra in enumerate(top_sutras_by_time, 1):
                    output.append(f"{i}. {sutra}: {avg_times[sutra]:.6f} seconds")
                output.append("")
                
                output.append("Top Sutras by Success Rate:")
                for i, sutra in enumerate(top_sutras_by_success, 1):
                    output.append(f"{i}. {sutra}: {success_rates[sutra]:.2f}%")
                output.append("")
                
                if top_interactions_by_freq:
                    output.append("Top Sutra Interactions by Frequency:")
                    for i, interaction in enumerate(top_interactions_by_freq, 1):
                        output.append(f"{i}. {' + '.join(interaction)}: {interaction_performance[interaction]['count']} occurrences")
                    output.append("")
                
                if top_interactions_by_time:
                    output.append("Top Sutra Interactions by Execution Time:")
                    for i, interaction in enumerate(top_interactions_by_time, 1):
                        output.append(f"{i}. {' + '.join(interaction)}: {interaction_performance[interaction]['avg_time']:.6f} seconds")
                
                return "\n".join(output)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in visualize_performance: {error_msg}")
            return f"Error generating performance visualization: {error_msg}"