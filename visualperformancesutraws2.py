from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def visualize_performance(self, n_top: int = 10, output_format: str = "matplotlib"):
    """Visualize sutra performance statistics stored on ``self``.

    Parameters
    ----------
    n_top : int
        Number of sutras or interactions to show.
    output_format : str
        Either ``"matplotlib"`` for graphical output or ``"text"`` for a plain
        string summary.
    """
    try:
        if not getattr(self, "performance_history", None):
            return "No performance data available."

        # Group data by sutra
        sutra_performance: Dict[str, Dict[str, List[float] | int]] = {}
        for record in self.performance_history:
            sutra = record["sutra"]
            success = record["success"]
            time = record["execution_time"]
            data = sutra_performance.setdefault(sutra, {
                "times": [],
                "success_count": 0,
                "failure_count": 0,
            })
            data["times"].append(time)
            if success:
                data["success_count"] += 1
            else:
                data["failure_count"] += 1

        avg_times = {
            s: sum(d["times"]) / len(d["times"])
            for s, d in sutra_performance.items() if d["times"]
        }
        success_rates = {
            s: d["success_count"] / (d["success_count"] + d["failure_count"]) * 100
            for s, d in sutra_performance.items() if (d["success_count"] + d["failure_count"]) > 0
        }

        top_by_time = sorted(avg_times.keys(), key=lambda s: avg_times[s])[:n_top]
        top_by_success = sorted(success_rates.keys(), key=lambda s: success_rates[s], reverse=True)[:n_top]

        if output_format == "text":
            lines = ["===== SUTRA PERFORMANCE ANALYSIS =====\n"]
            lines.append("Top Sutras by Execution Time:")
            for i, sutra in enumerate(top_by_time, 1):
                lines.append(f"{i}. {sutra}: {avg_times[sutra]:.6f} seconds")
            lines.append("")
            lines.append("Top Sutras by Success Rate:")
            for i, sutra in enumerate(top_by_success, 1):
                lines.append(f"{i}. {sutra}: {success_rates[sutra]:.2f}%")
            return "\n".join(lines)

        else:  # matplotlib output
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].barh(top_by_time, [avg_times[s] for s in top_by_time])
            axes[0].set_title("Top Sutras by Time")
            axes[0].set_xlabel("Average Execution Time (s)")
            axes[0].set_ylabel("Sutra")

            axes[1].barh(top_by_success, [success_rates[s] for s in top_by_success])
            axes[1].set_title("Top Sutras by Success Rate")
            axes[1].set_xlabel("Success Rate (%)")
            axes[1].set_ylabel("Sutra")

            plt.tight_layout()
            return fig
    except Exception as e:  # pragma: no cover - visualization errors
        logger.error("Error in visualize_performance: %s", e)
        return f"Error generating performance visualization: {e}"
