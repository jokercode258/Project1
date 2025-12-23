#!/usr/bin/env python
"""
VALIDATION SCRIPT
Kiểm tra toàn bộ hệ thống
"""

import sys
import os

def check_dependencies():
    """Kiểm tra các library cần thiết"""
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'chess': 'python-chess',
        'pygame': 'Pygame',
        'matplotlib': 'Matplotlib'
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name:20} installed")
        except ImportError:
            print(f"❌ {name:20} NOT installed")
            all_ok = False
    
    return all_ok

def check_files():
    """Kiểm tra các file Python"""
    print("\n" + "=" * 60)
    print("CHECKING FILES")
    print("=" * 60)
    
    files_to_check = {
        'board_state.py': 'State representation (BƯỚC 1)',
        'self_play.py': 'Self-play data generation (BƯỚC 2)',
        'value_network.py': 'Neural network (BƯỚC 3)',
        'train.py': 'Training loop (BƯỚC 4)',
        'minimax_engine.py': 'Minimax + Alpha-Beta (BƯỚC 5)',
        'gui.py': 'Pygame GUI (BƯỚC 6)',
        'main.py': 'Entry point',
        'requirements.txt': 'Dependencies',
        'README.md': 'Documentation'
    }
    
    all_ok = True
    for filename, description in files_to_check.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {filename:20} ({size:6d} bytes) - {description}")
        else:
            print(f"❌ {filename:20} MISSING - {description}")
            all_ok = False
    
    return all_ok

def check_imports():
    """Kiểm tra các imports"""
    print("\n" + "=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)
    
    test_imports = [
        ('from board_state import BoardState', 'BoardState'),
        ('from value_network import ValueNetwork', 'ValueNetwork'),
        ('from minimax_engine import MinimaxEngine', 'MinimaxEngine'),
        ('from self_play import SelfPlayManager', 'SelfPlayManager'),
        ('from train import ChessTrainer', 'ChessTrainer'),
        ('from gui import create_gui_with_engine', 'GUI'),
    ]
    
    all_ok = True
    for import_stmt, name in test_imports:
        try:
            exec(import_stmt)
            print(f"✅ {name:20} imports successfully")
        except Exception as e:
            print(f"❌ {name:20} import error: {e}")
            all_ok = False
    
    return all_ok

def check_basic_functionality():
    """Test các basic functionality"""
    print("\n" + "=" * 60)
    print("CHECKING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Test 1: Board state
        from board_state import BoardState
        import chess
        import numpy as np
        
        board = chess.Board()
        tensor = BoardState.board_to_tensor(board)
        assert tensor.shape == (12, 8, 8), "Board tensor shape incorrect"
        print("✅ Board state representation works")
        
        # Test 2: Network
        from value_network import ValueNetwork
        import torch
        
        network = ValueNetwork()
        dummy_input = torch.randn(2, 12, 8, 8)
        output = network(dummy_input)
        assert output.shape == (2, 1), "Network output shape incorrect"
        print("✅ Neural network forward pass works")
        
        # Test 3: Game
        from self_play import SelfPlayGame
        
        game = SelfPlayGame(max_moves=10)
        result, reason = game.play()
        assert result in [-1.0, 0.0, 1.0], "Game result invalid"
        print("✅ Self-play game works")
        
        # Test 4: Engine
        from minimax_engine import MinimaxEngine
        
        engine = MinimaxEngine(network, max_depth=2)
        move = engine.get_best_move(chess.Board())
        assert move is not None, "Minimax returned None"
        print("✅ Minimax engine works")
        
        return True
    except Exception as e:
        print(f"❌ Functionality check failed: {e}")
        return False

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  AI CHESS SYSTEM - VALIDATION SCRIPT".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Check dependencies
    deps_ok = check_dependencies()
    results.append(('Dependencies', deps_ok))
    
    # Check files
    files_ok = check_files()
    results.append(('Files', files_ok))
    
    # Check imports
    imports_ok = check_imports()
    results.append(('Imports', imports_ok))
    
    # Check functionality
    if deps_ok and files_ok and imports_ok:
        func_ok = check_basic_functionality()
        results.append(('Functionality', func_ok))
    else:
        print("\n⚠️  Skipping functionality check (dependencies/files/imports failed)")
        results.append(('Functionality', False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, status in results:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name:20} {'PASS' if status else 'FAIL'}")
    
    all_pass = all(status for _, status in results)
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL CHECKS PASSED! System is ready to use.")
        print("\nNext steps:")
        print("  1. python main.py train          # Train the model")
        print("  2. python main.py play           # Play vs AI")
        print("  3. python main.py analyze        # Analyze positions")
        print("\nFor more info: python main.py --help")
    else:
        print("❌ Some checks failed. Please fix issues above.")
        print("\nCommon solutions:")
        print("  • Install dependencies: pip install -r requirements.txt")
        print("  • Check Python version: python --version (3.8+ required)")
        print("  • Verify file permissions")
    print("=" * 60)
    
    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main())
