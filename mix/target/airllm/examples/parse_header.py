#!/usr/bin/env python3
"""
Example: Parse model header and show layer weight offsets

This demonstrates the model header parser and weight offset calculator.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airllm import parse_model_header, print_model_header
from airllm.weight_offsets import WeightOffsetCalculator


def main():
    """Demonstrate header parsing and offset calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse model header and show offsets')
    parser.add_argument('model_path', type=str,
                        help='Path to .m model file')
    parser.add_argument('--layer', type=int, default=0,
                        help='Layer to show detailed offsets (default: 0)')
    parser.add_argument('--max-seq-len', type=int, default=0,
                        help='Override maximum sequence length')
    
    args = parser.parse_args()
    
    # Parse model header
    print("=" * 60)
    print("PARSING MODEL HEADER")
    print("=" * 60)
    
    try:
        header = parse_model_header(args.model_path, args.max_seq_len)
        print_model_header(header)
    except Exception as e:
        print(f"Error parsing header: {e}")
        return 1
    
    # Calculate weight offsets
    print("\n" + "=" * 60)
    print("CALCULATING WEIGHT OFFSETS")
    print("=" * 60)
    
    offset_calc = WeightOffsetCalculator(header)
    
    # Show summary
    print(f"\nTotal layers: {header.n_layers}")
    print(f"Weight type: {header.weight_float_type.name}")
    
    for i in range(min(3, header.n_layers)):
        layer_offsets = offset_calc.get_layer_offsets(i)
        print(f"\nLayer {i}:")
        print(f"  Total size: {layer_offsets.total_size / 1024 / 1024:.2f} MB")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
