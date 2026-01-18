"""
Evaluation Visualization Script
================================
Generates plots and reports from evaluation results.

Usage:
    python plot_evaluation_results.py --eval-dir data/evaluation
    python plot_evaluation_results.py --csv data/evaluation/random/evaluation_log.csv --output random_plot.png
"""

import argparse
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib not installed. Install with: pip install matplotlib")


class EvaluationVisualizer:
    """
    Generate visualizations from evaluation results.
    """
    
    @staticmethod
    def read_csv_evaluation(csv_path: str) -> List[Dict]:
        """Read CSV evaluation log."""
        results = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    row['games'] = int(row['games'])
                    row['wins'] = int(row['wins'])
                    row['draws'] = int(row['draws'])
                    row['losses'] = int(row['losses'])
                    row['win_rate'] = float(row['win_rate'])
                    row['score'] = float(row['score'])
                    row['elo'] = float(row['elo'])
                    row['datetime'] = datetime.fromisoformat(row['timestamp'])
                    results.append(row)
        except Exception as e:
            print(f"[ERROR] Could not read CSV {csv_path}: {e}")
            return []
        return results
    
    @staticmethod
    def plot_elo_over_time(
        csv_path: str,
        output_path: str = "elo_over_time.png",
        title: str = "Elo Rating Over Time"
    ):
        """Plot Elo rating over time."""
        if not MATPLOTLIB_AVAILABLE:
            print("[ERROR] matplotlib required for plotting")
            return
        
        results = EvaluationVisualizer.read_csv_evaluation(csv_path)
        if not results:
            print(f"[ERROR] No data to plot from {csv_path}")
            return
        
        # Group by model
        by_model = {}
        for result in results:
            model_name = result.get('model_name', 'Unknown')
            if model_name not in by_model:
                by_model[model_name] = []
            by_model[model_name].append(result)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each model
        for model_name, evals in by_model.items():
            datetimes = [e['datetime'] for e in evals]
            elos = [e['elo'] for e in evals]
            ax.plot(datetimes, elos, marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('Timestamp', fontsize=12)
        ax.set_ylabel('Elo Rating', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"[INFO] Plot saved: {output_path}")
        plt.close()
    
    @staticmethod
    def plot_win_rate_comparison(
        csv_path: str,
        output_path: str = "win_rate_comparison.png",
        title: str = "Win Rate Comparison"
    ):
        """Plot win/draw/loss rates as stacked bar chart."""
        if not MATPLOTLIB_AVAILABLE:
            print("[ERROR] matplotlib required for plotting")
            return
        
        results = EvaluationVisualizer.read_csv_evaluation(csv_path)
        if not results:
            print(f"[ERROR] No data to plot from {csv_path}")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        labels = [f"{r['datetime'].strftime('%Y-%m-%d %H:%M')}\n({r['model_name'][:15]})" for r in results]
        win_rates = [r['win_rate'] * 100 for r in results]
        draw_rates = [r['draw_rate'] * 100 for r in results]
        loss_rates = [r['loss_rate'] * 100 for r in results]
        
        x = range(len(results))
        ax.bar(x, win_rates, label='Wins', color='green', alpha=0.7)
        ax.bar(x, draw_rates, bottom=win_rates, label='Draws', color='gray', alpha=0.7)
        ax.bar(x, loss_rates, bottom=[w + d for w, d in zip(win_rates, draw_rates)], 
               label='Losses', color='red', alpha=0.7)
        
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"[INFO] Plot saved: {output_path}")
        plt.close()
    
    @staticmethod
    def generate_text_report(
        eval_dir: str,
        output_path: str = "evaluation_report.txt"
    ):
        """Generate a text report from all evaluations."""
        report = []
        report.append("="*70)
        report.append("CHESS AI EVALUATION REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Scan for evaluation directories
        eval_path = Path(eval_dir)
        if not eval_path.exists():
            report.append(f"ERROR: Directory not found: {eval_dir}")
            return "\n".join(report)
        
        # Process each opponent directory
        for opponent_dir in eval_path.iterdir():
            if not opponent_dir.is_dir():
                continue
            
            report.append(f"\n{'='*70}")
            report.append(f"OPPONENT: {opponent_dir.name.upper()}")
            report.append(f"{'='*70}")
            
            # Find CSV file
            csv_file = opponent_dir / "evaluation_log.csv"
            if not csv_file.exists():
                report.append("No evaluation log found")
                continue
            
            results = EvaluationVisualizer.read_csv_evaluation(str(csv_file))
            if not results:
                report.append("No results in log")
                continue
            
            # Group by model
            by_model = {}
            for result in results:
                model_name = result.get('model_name', 'Unknown')
                if model_name not in by_model:
                    by_model[model_name] = []
                by_model[model_name].append(result)
            
            # Statistics for each model
            for model_name, evals in by_model.items():
                report.append(f"\nModel: {model_name}")
                report.append("-" * 70)
                
                avg_elo = sum(e['elo'] for e in evals) / len(evals)
                max_elo = max(e['elo'] for e in evals)
                min_elo = min(e['elo'] for e in evals)
                
                total_games = sum(e['games'] for e in evals)
                total_wins = sum(e['wins'] for e in evals)
                total_draws = sum(e['draws'] for e in evals)
                total_losses = sum(e['losses'] for e in evals)
                
                report.append(f"  Evaluations: {len(evals)}")
                report.append(f"  Total Games: {total_games}")
                report.append(f"  Overall Record: {total_wins}W {total_draws}D {total_losses}L")
                report.append(f"  Overall Win Rate: {100 * (total_wins + 0.5 * total_draws) / total_games:.1f}%")
                report.append(f"  Avg Elo: {avg_elo:.1f}")
                report.append(f"  Max Elo: {max_elo:.1f}")
                report.append(f"  Min Elo: {min_elo:.1f}")
                
                # Latest evaluation
                latest = evals[-1]
                report.append(f"\n  Latest Evaluation ({latest['datetime'].strftime('%Y-%m-%d %H:%M')})")
                report.append(f"    Games: {latest['games']}")
                report.append(f"    Record: {latest['wins']}W {latest['draws']}D {latest['losses']}L")
                report.append(f"    Win Rate: {latest['win_rate']*100:.1f}%")
                report.append(f"    Score: {latest['score']*100:.1f}%")
                report.append(f"    Elo: {latest['elo']:.1f}")
        
        report.append(f"\n{'='*70}")
        report_text = "\n".join(report)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"[INFO] Report saved: {output_path}")
        return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Visualize and report on evaluation results'
    )
    parser.add_argument(
        '--eval-dir',
        help='Evaluation directory (e.g., data/evaluation)',
        default='data/evaluation'
    )
    parser.add_argument(
        '--csv',
        help='CSV file to plot (e.g., data/evaluation/random/evaluation_log.csv)',
        default=None
    )
    parser.add_argument(
        '--output',
        help='Output file for plot',
        default=None
    )
    parser.add_argument(
        '--report',
        help='Generate text report',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    if args.csv:
        # Plot specific CSV
        if not Path(args.csv).exists():
            print(f"[ERROR] File not found: {args.csv}")
            return
        
        output = args.output or f"{Path(args.csv).stem}_elo.png"
        EvaluationVisualizer.plot_elo_over_time(args.csv, output)
        
        output2 = args.output or f"{Path(args.csv).stem}_winrate.png"
        EvaluationVisualizer.plot_win_rate_comparison(
            args.csv, 
            output2.replace('elo', 'winrate')
        )
    
    if args.report or (not args.csv and Path(args.eval_dir).exists()):
        # Generate report
        report_output = args.output or "evaluation_report.txt"
        report = EvaluationVisualizer.generate_text_report(args.eval_dir, report_output)
        print("\n" + report)


if __name__ == '__main__':
    main()
