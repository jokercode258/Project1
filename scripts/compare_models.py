"""
Model Comparison Script
=======================
Compare performance metrics across multiple model evaluations.

Usage:
    python compare_models.py --eval-dir data/evaluation
    python compare_models.py --opponent random --output comparison_report.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from chess_ai.evaluation_logger import EvaluationLogger


class ModelComparator:
    """
    Compare evaluation results across different models.
    """
    
    @staticmethod
    def load_json_evaluations(directory: str) -> List[Dict]:
        """Load all JSON evaluation files from directory."""
        evaluations = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return evaluations
        
        for json_file in dir_path.glob("eval_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    evaluations.append(data)
            except Exception as e:
                print(f"[WARNING] Could not load {json_file}: {e}")
        
        return evaluations
    
    @staticmethod
    def group_by_model(evaluations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group evaluations by model hash."""
        grouped = {}
        
        for eval_data in evaluations:
            model_hash = eval_data.get("model", {}).get("hash", "unknown")
            if model_hash not in grouped:
                grouped[model_hash] = []
            grouped[model_hash].append(eval_data)
        
        return grouped
    
    @staticmethod
    def generate_comparison_table(
        eval_dir: str,
        opponent: str = None,
        output_file: str = None
    ) -> str:
        """Generate a comparison table of models."""
        lines = []
        lines.append("="*100)
        lines.append("CHESS AI MODEL COMPARISON REPORT")
        lines.append("="*100)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        eval_path = Path(eval_dir)
        
        # Process each opponent
        for opponent_dir in eval_path.iterdir():
            if not opponent_dir.is_dir():
                continue
            
            if opponent and opponent.lower() != opponent_dir.name.lower():
                continue
            
            lines.append(f"\n{'='*100}")
            lines.append(f"OPPONENT: {opponent_dir.name.upper()}")
            lines.append(f"{'='*100}\n")
            
            evaluations = ModelComparator.load_json_evaluations(str(opponent_dir))
            if not evaluations:
                lines.append("No evaluations found.\n")
                continue
            
            # Group by model
            by_model = ModelComparator.group_by_model(evaluations)
            
            # Header
            lines.append(f"{'Model Name':<50} {'Games':>8} {'Record':>20} {'Score':>10} {'Elo':>10}")
            lines.append("-" * 100)
            
            # Sort by latest Elo (descending)
            sorted_models = sorted(
                by_model.items(),
                key=lambda x: x[1][-1].get("estimated_elo", 0),
                reverse=True
            )
            
            for rank, (model_hash, evals) in enumerate(sorted_models, 1):
                # Get latest evaluation
                latest = evals[-1]
                model_name = latest.get("model", {}).get("name", "Unknown")
                games = latest.get("results", {}).get("games", 0)
                wins = latest.get("results", {}).get("wins", 0)
                draws = latest.get("results", {}).get("draws", 0)
                losses = latest.get("results", {}).get("losses", 0)
                score = latest.get("results", {}).get("score", 0)
                elo = latest.get("estimated_elo", 0)
                
                record = f"{wins}W {draws}D {losses}L"
                
                lines.append(
                    f"{rank}. {model_name:<47} {games:>8} {record:>20} {score*100:>9.1f}% {elo:>10.0f}"
                )
            
            # Statistics
            if by_model:
                lines.append("\nSTATISTICS:")
                all_evals = [e for evals in by_model.values() for e in evals]
                avg_elo = sum(e.get("estimated_elo", 0) for e in all_evals) / len(all_evals)
                max_elo = max(e.get("estimated_elo", 0) for e in all_evals)
                min_elo = min(e.get("estimated_elo", 0) for e in all_evals)
                
                lines.append(f"  Total Models Evaluated: {len(by_model)}")
                lines.append(f"  Total Evaluations: {len(all_evals)}")
                lines.append(f"  Average Elo: {avg_elo:.1f}")
                lines.append(f"  Max Elo: {max_elo:.1f}")
                lines.append(f"  Min Elo: {min_elo:.1f}")
        
        lines.append(f"\n{'='*100}\n")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"[INFO] Report saved: {output_file}")
        
        return report
    
    @staticmethod
    def generate_detailed_model_report(
        model_path: str,
        eval_dirs: List[str],
        output_file: str = None
    ) -> str:
        """Generate detailed report for a specific model."""
        lines = []
        lines.append("="*80)
        lines.append("DETAILED MODEL EVALUATION REPORT")
        lines.append("="*80)
        lines.append(f"Model: {Path(model_path).name}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Find evaluations for this model
        for eval_dir in eval_dirs:
            eval_path = Path(eval_dir)
            if not eval_path.exists():
                continue
            
            evaluations = ModelComparator.load_json_evaluations(str(eval_path))
            model_evals = [
                e for e in evaluations
                if e.get("model", {}).get("name", "") == Path(model_path).name
            ]
            
            if not model_evals:
                continue
            
            opponent = eval_path.name.upper()
            lines.append(f"\n{'='*80}")
            lines.append(f"OPPONENT: {opponent}")
            lines.append(f"{'='*80}\n")
            
            # Sort by timestamp
            model_evals.sort(key=lambda x: x.get("timestamp", ""))
            
            for i, eval_data in enumerate(model_evals, 1):
                lines.append(f"Evaluation #{i}")
                lines.append(f"  Timestamp: {eval_data.get('datetime_readable', 'Unknown')}")
                lines.append(f"  Games: {eval_data.get('results', {}).get('games', 0)}")
                
                results = eval_data.get("results", {})
                wins = results.get("wins", 0)
                draws = results.get("draws", 0)
                losses = results.get("losses", 0)
                lines.append(f"  Record: {wins}W {draws}D {losses}L")
                lines.append(f"  Win Rate: {results.get('win_rate', 0)*100:.1f}%")
                lines.append(f"  Score: {results.get('score', 0)*100:.1f}%")
                lines.append(f"  Elo: {eval_data.get('estimated_elo', 0):.1f}")
                
                # Extra info
                extra = eval_data.get("extra", {})
                if extra:
                    lines.append(f"  Extra Info:")
                    for key, value in extra.items():
                        if value is not None:
                            lines.append(f"    - {key}: {value}")
                
                lines.append("")
        
        lines.append("="*80)
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"[INFO] Report saved: {output_file}")
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description='Compare Chess AI model evaluations'
    )
    parser.add_argument(
        '--eval-dir',
        help='Root evaluation directory (default: data/evaluation)',
        default='data/evaluation'
    )
    parser.add_argument(
        '--opponent',
        help='Filter by opponent (e.g., random, stockfish)',
        default=None
    )
    parser.add_argument(
        '--model',
        help='Path to specific model for detailed report',
        default=None
    )
    parser.add_argument(
        '--output',
        help='Output file path',
        default=None
    )
    
    args = parser.parse_args()
    
    if args.model:
        # Detailed model report
        eval_subdirs = [
            str(Path(args.eval_dir) / d) 
            for d in ['random', 'stockfish'] 
            if (Path(args.eval_dir) / d).exists()
        ]
        report = ModelComparator.generate_detailed_model_report(
            args.model,
            eval_subdirs,
            args.output
        )
    else:
        # Comparison table
        report = ModelComparator.generate_comparison_table(
            args.eval_dir,
            args.opponent,
            args.output
        )
    
    print(report)


if __name__ == '__main__':
    main()
