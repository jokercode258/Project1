import json
import csv
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class EvaluationLogger:
    """
    Centralized evaluation logging system for Chess AI experiments.
    """
    
    @staticmethod
    def get_model_hash(model_path: str, length: int = 8) -> str:
        """
        Generate a short hash of the model file for tracking.
        
        Args:
            model_path: Path to the model file
            length: Length of hash string (default: 8)
        
        Returns:
            Short hash string
        """
        try:
            with open(model_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:length]
            return file_hash
        except Exception as e:
            print(f"[WARNING] Could not hash model: {e}")
            return "unknown"
    
    @staticmethod
    def save_json_result(
        output_dir: str,
        model_path: str,
        opponent: str,
        games: int,
        wins: int,
        draws: int,
        losses: int,
        estimated_elo: float,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> str:

        os.makedirs(output_dir, exist_ok=True)
        
        model_name = Path(model_path).name if model_path else "default"
        model_hash = EvaluationLogger.get_model_hash(model_path) if model_path else "unknown"
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "datetime_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": {
                "path": str(model_path),
                "name": model_name,
                "hash": model_hash
            },
            "opponent": opponent,
            "results": {
                "games": games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": round(wins / games if games > 0 else 0, 4),
                "draw_rate": round(draws / games if games > 0 else 0, 4),
                "loss_rate": round(losses / games if games > 0 else 0, 4),
                "score": round((wins + 0.5 * draws) / games if games > 0 else 0, 4)
            },
            "estimated_elo": round(estimated_elo, 2),
            "extra": extra_info or {}
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{opponent.lower()}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Evaluation result saved: {filepath}")
        return filepath
    
    @staticmethod
    def append_csv_record(
        output_dir: str,
        csv_filename: str,
        model_path: str,
        opponent: str,
        games: int,
        wins: int,
        draws: int,
        losses: int,
        estimated_elo: float,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> str:

        os.makedirs(output_dir, exist_ok=True)
        
        model_name = Path(model_path).name if model_path else "default"
        model_hash = EvaluationLogger.get_model_hash(model_path) if model_path else "unknown"
        
        csv_path = os.path.join(output_dir, csv_filename)
        file_exists = os.path.exists(csv_path)
        
        row = {
            "timestamp": datetime.now().isoformat(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "model_hash": model_hash,
            "opponent": opponent,
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": round(wins / games if games > 0 else 0, 4),
            "draw_rate": round(draws / games if games > 0 else 0, 4),
            "score": round((wins + 0.5 * draws) / games if games > 0 else 0, 4),
            "elo": round(estimated_elo, 2),
        }
        
        # Add extra info columns
        if extra_info:
            for key, value in extra_info.items():
                row[f"extra_{key}"] = value
        
        fieldnames = list(row.keys())
        
        with open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        print(f"[INFO] CSV record appended: {csv_path}")
        return csv_path
    
    @staticmethod
    def generate_comparison_report(
        json_files: list,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:

        results = []
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"[WARNING] Could not read {json_file}: {e}")
        
        # Group by model
        by_model = {}
        for result in results:
            model_hash = result.get("model", {}).get("hash", "unknown")
            if model_hash not in by_model:
                by_model[model_hash] = []
            by_model[model_hash].append(result)
        
        # Calculate statistics
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_evaluations": len(results),
            "models": {}
        }
        
        for model_hash, evals in by_model.items():
            model_data = {
                "evaluations": len(evals),
                "opponents": list(set(e.get("opponent", "") for e in evals)),
                "avg_elo": round(sum(e.get("estimated_elo", 0) for e in evals) / len(evals), 2),
                "latest": evals[-1] if evals else None
            }
            report["models"][model_hash] = model_data
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Comparison report saved: {output_path}")
        
        return report


# Convenience functions for backward compatibility
def save_evaluation_result(
    output_dir: str,
    model_path: str,
    opponent: str,
    games: int,
    wins: int,
    draws: int,
    losses: int,
    estimated_elo: float,
    extra_info: Optional[Dict[str, Any]] = None
):
    """
    Simple wrapper for saving evaluation results.
    (For backward compatibility)
    """
    EvaluationLogger.save_json_result(
        output_dir=output_dir,
        model_path=model_path,
        opponent=opponent,
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        estimated_elo=estimated_elo,
        extra_info=extra_info
    )
    
    EvaluationLogger.append_csv_record(
        output_dir=output_dir,
        csv_filename="evaluation_log.csv",
        model_path=model_path,
        opponent=opponent,
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        estimated_elo=estimated_elo,
        extra_info=extra_info
    )
