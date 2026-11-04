"""HI 744 retrieval component."""
from pathlib import Path
import json


def load_patients(folder):
    paths = sorted(Path(folder).rglob('*.json'))
    patients = []
    for path in paths:
        if path.name.startswith('._') or '__MACOSX' in path.parts:
            continue
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError):
            continue
        if isinstance(obj, dict) and 'patient_uid' in obj and 'patient' in obj:
            patients.append(obj)
    if not patients:
        raise ValueError(f'No patient JSON records found in {folder}')
    ids = [str(p['patient_uid']) for p in patients]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate patient_uid detected')
    return patients
