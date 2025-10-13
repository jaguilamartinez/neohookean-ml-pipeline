import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
from pathlib import Path


class Preprocessor:
    def __init__(self, test_size=0.2, val_size=0.2, scaler='standard', random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        if scaler == 'standard':
            self.X_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
        elif scaler == 'minmax':
            self.X_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler}")
        
        self.fitted = False
    
    def split_data(self, X, y, case_numbers=None):
        # First split: train+val vs test
        if case_numbers is not None:
            print("\nPerforming case-based data splitting...")
            unique_cases = np.unique(case_numbers)
            n_test_cases = max(1, int(len(unique_cases) * self.test_size))
            
            print(f"  Total cases: {len(unique_cases)}")
            print(f"  Test cases: {n_test_cases} ({n_test_cases/len(unique_cases)*100:.1f}%)")
            
            np.random.seed(self.random_state)
            test_cases = np.random.choice(unique_cases, size=n_test_cases, replace=False)
            
            test_mask = np.isin(case_numbers, test_cases)
            train_val_mask = ~test_mask
            
            X_train_val, X_test = X[train_val_mask], X[test_mask]
            y_train_val, y_test = y[train_val_mask], y[test_mask]
            case_train_val = case_numbers[train_val_mask]
            
            print(f"  Test data points: {X_test.shape[0]}")
        else:
            print("\nPerforming random data splitting (no case information)...")
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            case_train_val = None
        
        # Second split: train vs val
        val_size_adj = self.val_size / (1 - self.test_size)
        
        if case_train_val is not None:
            unique_train_cases = np.unique(case_train_val)
            n_val_cases = max(1, int(len(unique_train_cases) * val_size_adj))
            n_train_cases = len(unique_train_cases) - n_val_cases
            
            print(f"  Train cases: {n_train_cases} ({n_train_cases/len(unique_train_cases)*100:.1f}%)")
            print(f"  Validation cases: {n_val_cases} ({n_val_cases/len(unique_train_cases)*100:.1f}%)")
            
            np.random.seed(self.random_state + 1)
            val_cases = np.random.choice(unique_train_cases, size=n_val_cases, replace=False)
            
            val_mask = np.isin(case_train_val, val_cases)
            train_mask = ~val_mask
            
            X_train, X_val = X_train_val[train_mask], X_train_val[val_mask]
            y_train, y_val = y_train_val[train_mask], y_train_val[val_mask]
            
            print(f"  Final splits - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size_adj, random_state=self.random_state
            )
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
    
    def fit_scalers(self, X_train, y_train):
        self.X_scaler.fit(X_train)
        self.y_scaler.fit(y_train)
        self.fitted = True
    
    def transform(self, data):
        if not self.fitted:
            raise ValueError("Scalers not fitted")
        
        result = {}
        for key in ['train', 'val', 'test']:
            X_key, y_key = f'X_{key}', f'y_{key}'
            if X_key in data and y_key in data:
                result[f'{X_key}_scaled'] = self.X_scaler.transform(data[X_key])
                result[f'{y_key}_scaled'] = self.y_scaler.transform(data[y_key])
                result[X_key] = data[X_key]
                result[y_key] = data[y_key]
        
        return result
    
    def validate_case_split(self, X, y, case_numbers, splits):
        """Validate that no cases are split across train/val/test sets."""
        print("\nValidating case-based split integrity...")
        
        # Get case numbers for each split
        train_mask = np.isin(np.arange(len(X)), [i for i in range(len(X)) if any(np.array_equal(X[i], row) for row in splits['X_train'])])
        val_mask = np.isin(np.arange(len(X)), [i for i in range(len(X)) if any(np.array_equal(X[i], row) for row in splits['X_val'])])
        test_mask = np.isin(np.arange(len(X)), [i for i in range(len(X)) if any(np.array_equal(X[i], row) for row in splits['X_test'])])
        
        train_cases = set(case_numbers[train_mask])
        val_cases = set(case_numbers[val_mask]) 
        test_cases = set(case_numbers[test_mask])
        
        # Check for overlaps
        train_val_overlap = train_cases.intersection(val_cases)
        train_test_overlap = train_cases.intersection(test_cases)
        val_test_overlap = val_cases.intersection(test_cases)
        
        print(f"  Train cases: {len(train_cases)}")
        print(f"  Validation cases: {len(val_cases)}")
        print(f"  Test cases: {len(test_cases)}")
        
        if train_val_overlap:
            print(f"  Train-Val overlap: {len(train_val_overlap)} cases")
        else:
            print(f"  No Train-Val case overlap")
            
        if train_test_overlap:
            print(f"  Train-Test overlap: {len(train_test_overlap)} cases")
        else:
            print(f"  No Train-Test case overlap")
            
        if val_test_overlap:
            print(f"  Val-Test overlap: {len(val_test_overlap)} cases")
        else:
            print(f"  No Val-Test case overlap")
            
        total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
        if total_overlap == 0:
            print(f"  Case-based splitting successful - no data leakage detected!")
        else:
            print(f"  WARNING: {total_overlap} case overlaps detected - data leakage risk!")
    
    def inverse_transform_y(self, y_scaled):
        return self.y_scaler.inverse_transform(y_scaled)
    
    def preprocess(self, X, y, case_numbers=None):
        splits = self.split_data(X, y, case_numbers)
        self.fit_scalers(splits['X_train'], splits['y_train'])
        return self.transform(splits)
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.X_scaler, path / 'X_scaler.pkl')
        joblib.dump(self.y_scaler, path / 'y_scaler.pkl')
        
        config = {
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'fitted': self.fitted
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path):
        path = Path(path)
        
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        
        preprocessor = cls(
            test_size=config['test_size'],
            val_size=config['val_size'],
            random_state=config['random_state']
        )
        
        preprocessor.X_scaler = joblib.load(path / 'X_scaler.pkl')
        preprocessor.y_scaler = joblib.load(path / 'y_scaler.pkl')
        preprocessor.fitted = config['fitted']
        
        return preprocessor