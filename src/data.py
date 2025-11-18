"""Data loading and preprocessing."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings


STRAIN_COMPONENTS = ['E_xx', 'E_yy', 'γ_xy']
STRESS_COMPONENTS = ['S_xx', 'S_yy', 'S_xy']


def _apply_scaling(X_train, X_val, X_test, y_train, y_val, y_test, scale):
    """Apply scaling to train/val/test splits if requested."""
    if scale:
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train_out = X_scaler.fit_transform(X_train)
        X_val_out = X_scaler.transform(X_val)
        X_test_out = X_scaler.transform(X_test)
        
        y_train_out = y_scaler.fit_transform(y_train)
        y_val_out = y_scaler.transform(y_val)
        y_test_out = y_scaler.transform(y_test)
    else:
        X_scaler = None
        y_scaler = None
        X_train_out = X_train
        X_val_out = X_val
        X_test_out = X_test
        y_train_out = y_train
        y_val_out = y_val
        y_test_out = y_test
    
    return X_train_out, X_val_out, X_test_out, y_train_out, y_val_out, y_test_out, X_scaler, y_scaler


def load_dataset(data_path):
    """Load dataset from neohookean-data-generator.
    
    Returns dict with strains, stresses, case_ids, scenario_labels.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    data = np.load(data_path)
    
    strains = data['strains']
    stresses = data['stresses']
    
    # Data generator already provides Voigt notation [E11, E22, gamma12]
    # where gamma12 = 2*E12, so no conversion needed
    strains_voigt = strains
    
    dataset = {
        'strains': strains_voigt,
        'stresses': stresses,
        'case_ids': data.get('case_ids', None),
        'scenario_labels': data.get('scenario_labels', None),
        'n_samples': strains.shape[0]
    }
    
    return dataset


def prepare_data(strains, stresses, test_size=0.2, val_size=0.2, random_state=42, scale=False):
    """Split data into train/val/test sets with optional scaling.
    
    scale=True for baseline model, scale=False for PANN (requires physical values).
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        strains, stresses, test_size=test_size, random_state=random_state
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
    )
    
    X_train_out, X_val_out, X_test_out, y_train_out, y_val_out, y_test_out, X_scaler, y_scaler = \
        _apply_scaling(X_train, X_val, X_test, y_train, y_val, y_test, scale)
    
    return {
        'X_train': X_train_out,
        'X_val': X_val_out,
        'X_test': X_test_out,
        'y_train': y_train_out,
        'y_val': y_val_out,
        'y_test': y_test_out,
        'y_test_original': y_test,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'splits': {
            'train': X_train.shape[0],
            'val': X_val.shape[0],
            'test': X_test.shape[0]
        }
    }


def prepare_data_by_groups(strains, stresses, group_ids, test_size=0.2, val_size=0.2, 
                           random_state=42, scale=False, split_by='case'):
    """Split by groups to prevent data leakage.
    
    Keeps all samples from same group together in one split.
    """
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    
    print(f"Splitting by {split_by}: {n_groups} unique groups")
    
    groups_train_val, groups_test = train_test_split(
        unique_groups, test_size=test_size, random_state=random_state
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    groups_train, groups_val = train_test_split(
        groups_train_val, test_size=val_size_adjusted, random_state=random_state
    )
    
    train_mask = np.isin(group_ids, groups_train)
    val_mask = np.isin(group_ids, groups_val)
    test_mask = np.isin(group_ids, groups_test)
    
    X_train = strains[train_mask]
    X_val = strains[val_mask]
    X_test = strains[test_mask]
    
    y_train = stresses[train_mask]
    y_val = stresses[val_mask]
    y_test = stresses[test_mask]
    
    train_groups = group_ids[train_mask]
    val_groups = group_ids[val_mask]
    test_groups = group_ids[test_mask]
    
    # Validate no leakage
    train_set = set(groups_train)
    val_set = set(groups_val)
    test_set = set(groups_test)
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(f"Group leakage detected! Train/Val: {len(overlap_train_val)}, "
                        f"Train/Test: {len(overlap_train_test)}, Val/Test: {len(overlap_val_test)}")
    
    print(f"  Train: {len(groups_train)} groups, {len(X_train)} samples")
    print(f"  Val:   {len(groups_val)} groups, {len(X_val)} samples")
    print(f"  Test:  {len(groups_test)} groups, {len(X_test)} samples")
    print(f"  ✓ No group leakage detected")
    
    X_train_out, X_val_out, X_test_out, y_train_out, y_val_out, y_test_out, X_scaler, y_scaler = \
        _apply_scaling(X_train, X_val, X_test, y_train, y_val, y_test, scale)
    
    return {
        'X_train': X_train_out,
        'X_val': X_val_out,
        'X_test': X_test_out,
        'y_train': y_train_out,
        'y_val': y_val_out,
        'y_test': y_test_out,
        'y_test_original': y_test,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'splits': {
            'train': X_train.shape[0],
            'val': X_val.shape[0],
            'test': X_test.shape[0]
        },
        'groups': {
            'train': groups_train,
            'val': groups_val,
            'test': groups_test,
            'train_group_ids': train_groups,
            'val_group_ids': val_groups,
            'test_group_ids': test_groups
        }
    }


def validate_no_leakage(data_dict):
    """Validate no group leakage between splits."""
    if 'groups' not in data_dict:
        warnings.warn("No group information found. Cannot validate leakage. "
                     "Use prepare_data_by_groups() instead of prepare_data().")
        return False
    
    groups = data_dict['groups']
    train_set = set(groups['train'])
    val_set = set(groups['val'])
    test_set = set(groups['test'])
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    if overlap_train_val:
        raise ValueError(f"Group leakage: {len(overlap_train_val)} groups in both train and val")
    if overlap_train_test:
        raise ValueError(f"Group leakage: {len(overlap_train_test)} groups in both train and test")
    if overlap_val_test:
        raise ValueError(f"Group leakage: {len(overlap_val_test)} groups in both val and test")
    
    print("✓ No group leakage between train/val/test splits")
    return True


def get_component_names():
    """Return strain and stress component names in Voigt notation."""
    return STRAIN_COMPONENTS, STRESS_COMPONENTS
