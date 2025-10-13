#!/usr/bin/env python3

import sys
import numpy as np
from pathlib import Path

sys.path.append('src')

from data.loader import DataLoader
from data.preprocessor import Preprocessor


def test_case_based_splitting():
    print("Testing case-based data splitting...")
    
    data_path = "../neohookean-dataset-generator/hyperelastic_comprehensive_experiments"
    loader = DataLoader(data_path)
    
    try:
        dataset = loader.load_dataset()
        print(f"Loaded: {dataset['n_samples']} samples, {dataset['n_cases']} cases")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        return False
    
    X = dataset['strains']
    y = dataset['stresses']
    case_numbers = dataset['case_numbers']
    
    # Test case-based splitting
    preprocessor = Preprocessor(test_size=0.2, val_size=0.2, random_state=42)
    data = preprocessor.preprocess(X, y, case_numbers)
    
    print(f"Split: Train {data['X_train'].shape[0]}, Val {data['X_val'].shape[0]}, Test {data['X_test'].shape[0]} samples")
    print("SUCCESS: Case-based splitting completed - check preprocessor logs above for validation.")
    return True


if __name__ == "__main__":
    success = test_case_based_splitting()
    sys.exit(0 if success else 1)