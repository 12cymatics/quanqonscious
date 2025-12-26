def reset_performance_tracking(self):
    """Resets all performance tracking data"""
    self.performance_history = []
    self.sutra_interactions = {}
        
def get_performance_summary(self) -> Dict[str, Any]:
    """Returns a summary of sutra performance statistics."""
    if not self.performance_history:
        return {"error": "No performance data available."}
            
        summary = {
            "total_executions": len(self.performance_history),
            "success_rate": sum(1 for r in self.performance_history if r['success']) / len(self.performance_history) * 100,
            "avg_execution_time": sum(r['execution_time'] for r in self.performance_history) / len(self.performance_history),
        "sutra_stats": {},
            "interaction_stats": {}
        }
        
        # Compute sutra-specific statistics
        sutra_data = {}
        for record in self.performance_history:
            sutra = record['sutra']
            if sutra not in sutra_data:
                sutra_data[sutra] = {
                    "count": 0,
                    "success_count": 0,
                    "total_time": 0.0
                }
            
            sutra_data[sutra]["count"] += 1
            if record['success']:
                sutra_data[sutra]["success_count"] += 1
            sutra_data[sutra]["total_time"] += record['execution_time']
        
        # Calculate averages and success rates
        for sutra, data in sutra_data.items():
            summary["sutra_stats"][sutra] = {
                "execution_count": data["count"],
                "success_rate": data["success_count"] / data["count"] * 100,
                "avg_execution_time": data["total_time"] / data["count"]
            }
        
        # Interaction statistics
        for interaction, data in self.sutra_interactions.items():
            summary["interaction_stats"][" + ".join(interaction)] = {
                "count": data["count"],
                "avg_execution_time": data["avg_execution_time"]
            }
        
        return summary