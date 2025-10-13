import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.consolidated_file = self.data_path / "consolidated_dataset.npz"
        self.ml_ready_file = self.data_path / "ml_ready_dataset.npz"
        self.training_file = self.data_path / "training_data.npz"
    
    def load_dataset(self):
        # Try ML-ready dataset first (comprehensive experiments)
        if self.ml_ready_file.exists():
            print(f"Loading ML-ready dataset: {self.ml_ready_file}")
            data = np.load(self.ml_ready_file)
            
            # Create strain type array for all data points
            strain_types = None
            if 'case_strain_types' in data and 'unique_case_numbers' in data:
                case_to_strain_type = dict(zip(data['unique_case_numbers'], data['case_strain_types']))
                strain_types = np.array([case_to_strain_type[case] for case in data['case_numbers']])
            
            dataset = {
                'strains': data['strains'],
                'stresses': data['stresses'], 
                'case_numbers': data['case_numbers'],
                'n_cases': int(data['n_cases']),
                'n_samples': int(data['n_samples']),
                'strain_components': data['strain_components'],
                'stress_components': data['stress_components']
            }
            
            if strain_types is not None:
                dataset['strain_types'] = strain_types
            
            return dataset
        # Fallback to consolidated dataset (legacy)
        elif self.consolidated_file.exists():
            print(f"Loading consolidated dataset: {self.consolidated_file}")
            data = np.load(self.consolidated_file)
            return {
                'strains': data['strains'],
                'stresses': data['stresses'],
                'case_numbers': data['case_numbers'],
                'n_cases': int(data['n_cases']),
                'n_samples': int(data['n_samples']),
                'strain_components': data['strain_components'],
                'stress_components': data['stress_components']
            }
        else:
            raise FileNotFoundError(f"No dataset found. Tried: {self.ml_ready_file}, {self.consolidated_file}")
    
    def load_training_data(self):
        if not self.training_file.exists():
            raise FileNotFoundError(f"Training data not found: {self.training_file}")
        
        data = np.load(self.training_file)
        return data['X'], data['y']
    
    def validate_case_integrity(self, dataset):
        """Validate that case numbers are properly structured for case-based splitting."""
        case_numbers = dataset['case_numbers']
        unique_cases = np.unique(case_numbers)
        
        print(f"\nCase-based split validation:")
        print(f"  Total data points: {len(case_numbers)}")
        print(f"  Total cases: {len(unique_cases)}")
        print(f"  Case range: {unique_cases.min()} to {unique_cases.max()}")
        
        # Check for consistent case sizes
        case_sizes = []
        for case in unique_cases[:5]:  # Check first 5 cases
            case_size = np.sum(case_numbers == case)
            case_sizes.append(case_size)
        
        print(f"  Steps per case (first 5): {case_sizes}")
        
        if len(set(case_sizes)) == 1:
            print(f"  Consistent case structure: {case_sizes[0]} steps per case")
        else:
            print(f"  Inconsistent case sizes detected")
        
        return len(unique_cases), case_sizes[0] if case_sizes else 0
    
    def to_dataframe(self, dataset):
        strain_cols = [f'strain_{comp.lower()}' for comp in dataset['strain_components']]
        stress_cols = [f'stress_{comp.lower()}' for comp in dataset['stress_components']]
        
        df_data = [dataset['strains'], dataset['stresses'], dataset['case_numbers']]
        columns = strain_cols + stress_cols + ['case_number']
        
        # Add strain types if available
        if 'strain_types' in dataset:
            df_data.append(dataset['strain_types'])
            columns.append('strain_type')
        
        df_data = np.column_stack(df_data)
        df = pd.DataFrame(df_data, columns=columns)
        df['case_number'] = df['case_number'].astype(int)
        
        return df
