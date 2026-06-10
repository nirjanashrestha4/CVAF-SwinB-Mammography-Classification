"""Dataset pipeline for CBIS-DDSM and VinDr-Mammo.

Binary labels: 0 = Benign, 1 = Malignant.
Each sample is a complete four-view case (L-CC, L-MLO, R-CC, R-MLO).
The split is patient-level and stratified (70 percent train, 15 percent
validation, 15 percent test); all four views of a patient stay together.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


# BI-RADS 1-3 map to Benign, 4-5 map to Malignant, 0 is excluded.
VINDR_BIRADS_MAP = {0: None, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1}

REQUIRED_VIEWS = {('LEFT', 'CC'), ('LEFT', 'MLO'),
                  ('RIGHT', 'CC'), ('RIGHT', 'MLO')}


def get_transforms(img_size: int = 224, split: str = 'train') -> Callable:
    """Return the image transform pipeline for a split.

    Training applies augmentation; validation and test apply resize and
    normalisation only so evaluation is reproducible.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == 'train':
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class FourViewDataset(Dataset):
    """Loads all four mammographic views per patient.

    Returns a dict with keys lcc, lmlo, rcc, rmlo and a float label.
    """

    def __init__(self, samples: List[dict], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        views = {}
        for key in ('lcc', 'lmlo', 'rcc', 'rmlo'):
            img = Image.open(s[key]).convert('RGB')
            views[key] = self.transform(img)
        label = torch.tensor(s['label'], dtype=torch.float32)
        return views, label


def _stratified_split(
    samples: List[dict],
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split samples 70/15/15, stratified by label where possible."""
    labels = [s['label'] for s in samples]
    n_min = min(labels.count(0), labels.count(1))
    stratify1 = labels if n_min >= 6 else None

    train_s, temp_s, _, temp_l = train_test_split(
        samples, labels,
        test_size=0.30,
        stratify=stratify1,
        random_state=seed,
    )
    stratify2 = temp_l if min(temp_l.count(0), temp_l.count(1)) >= 2 else None
    val_s, test_s = train_test_split(
        temp_s,
        test_size=0.50,
        stratify=stratify2,
        random_state=seed,
    )
    return train_s, val_s, test_s


def _build_cbis_samples(cbis_root: str) -> List[dict]:
    """Build four-view samples from CBIS-DDSM.

    Reads the mass and calcification case description CSVs, keeps only full
    mammogram images via dicom_info.csv, and groups complete four-view cases.
    """
    root = Path(cbis_root)
    csv_dir = root / 'csv'
    jpg_dir = root / 'jpeg'

    if not csv_dir.exists():
        print(f'[CBIS-DDSM] csv/ not found under {root}')
        return []

    csv_files = [
        'mass_case_description_train_set.csv',
        'mass_case_description_test_set.csv',
        'calc_case_description_train_set.csv',
        'calc_case_description_test_set.csv',
    ]
    dfs = []
    for csv_file in csv_files:
        p = csv_dir / csv_file
        if p.exists():
            dfs.append(pd.read_csv(p))
            print(f'[CBIS-DDSM] Loaded {csv_file}  ({len(dfs[-1])} rows)')
    if not dfs:
        print('[CBIS-DDSM] No case CSV files found')
        return []

    df = pd.concat(dfs, ignore_index=True)
    print(f'[CBIS-DDSM] Combined: {len(df)} rows')

    if not jpg_dir.exists():
        print('[CBIS-DDSM] jpeg/ folder not found')
        return []

    print('[CBIS-DDSM] Indexing JPEG images ...')
    img_index: Dict[str, str] = {}
    for p in jpg_dir.rglob('*.jpg'):
        img_index[p.parent.name] = str(p)
    print(f'[CBIS-DDSM] Indexed {len(img_index):,} JPEG files')

    dicom_csv = csv_dir / 'dicom_info.csv'
    full_mammo_index: Dict[str, str] = {}
    if dicom_csv.exists():
        ddf = pd.read_csv(dicom_csv)
        full = ddf[ddf['SeriesDescription'] == 'full mammogram images']
        print(f'[CBIS-DDSM] Full mammogram series: {len(full)}')
        for _, row in full.iterrows():
            img_path = str(row.get('image_path', ''))
            parts = img_path.replace('\\', '/').replace('\n', '').strip().split('/')
            if len(parts) >= 3:
                series_uid = parts[-2]
                full_path = jpg_dir / series_uid
                if full_path.exists():
                    files = [f for f in os.listdir(str(full_path))
                             if f.endswith('.jpg')]
                    if files:
                        full_mammo_index[series_uid] = str(full_path / files[0])
        print(f'[CBIS-DDSM] Full mammogram images indexed: {len(full_mammo_index)}')
    else:
        print('[CBIS-DDSM] dicom_info.csv not found, using all images')
        full_mammo_index = img_index

    df['side'] = df['left or right breast'].str.upper().str.strip()
    df['view'] = df['image view'].str.upper().str.strip()
    df['pid'] = df['patient_id'].str.strip()
    df['label'] = df['pathology'].str.upper().str.strip().map(
        {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}
    )
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    def resolve_path(file_path):
        if pd.isna(file_path):
            return None
        parts = str(file_path).replace('\n', '').strip().split('/')
        if len(parts) >= 2:
            series_uid = parts[-2]
            if series_uid in full_mammo_index:
                return full_mammo_index[series_uid]
        return None

    df['img_path'] = df['image file path'].apply(resolve_path)
    df = df.dropna(subset=['img_path'])
    print(f'[CBIS-DDSM] Rows with resolved paths: {len(df)}')

    samples_d: Dict[str, dict] = {}
    for _, row in df.iterrows():
        pid = row['pid']
        key = (row['side'], row['view'])
        if key not in REQUIRED_VIEWS:
            continue
        if pid not in samples_d:
            samples_d[pid] = {'label': 0, 'paths': {}}
        samples_d[pid]['paths'][key] = row['img_path']
        samples_d[pid]['label'] = max(samples_d[pid]['label'], row['label'])

    samples = []
    for pid, data in samples_d.items():
        if not REQUIRED_VIEWS.issubset(data['paths'].keys()):
            continue
        samples.append({
            'patient_id': pid,
            'label': data['label'],
            'lcc': data['paths'][('LEFT', 'CC')],
            'lmlo': data['paths'][('LEFT', 'MLO')],
            'rcc': data['paths'][('RIGHT', 'CC')],
            'rmlo': data['paths'][('RIGHT', 'MLO')],
            'source': 'cbis',
        })

    n_b = sum(1 for s in samples if s['label'] == 0)
    n_m = sum(1 for s in samples if s['label'] == 1)
    print(f'[CBIS-DDSM] Complete 4-view cases: {len(samples)}  '
          f'Benign={n_b}  Malignant={n_m}')
    return samples


def _build_vindr_samples(
    root: str,
    vindr_images: Optional[str] = None,
    vindr_labels: Optional[str] = None,
) -> List[dict]:
    """Build four-view samples from VinDr-Mammo.

    Parses BI-RADS (integer or "BI-RADS n" string), normalises laterality,
    and groups complete four-view studies.
    """
    labels_dir = Path(vindr_labels) if vindr_labels else Path(root)
    images_dir = Path(vindr_images) if vindr_images else Path(root)
    ann_path = labels_dir / 'breast-level_annotations.csv'

    if not ann_path.exists():
        raise FileNotFoundError(f'VinDr annotations not found: {ann_path}')

    df = pd.read_csv(ann_path)
    print(f'[VinDr-Mammo] Loaded breast-level_annotations.csv  ({len(df):,} rows)')

    def parse_birads(val):
        if pd.isna(val):
            return None
        s = str(val).upper().replace('BI-RADS', '').strip()
        try:
            return VINDR_BIRADS_MAP.get(int(s))
        except ValueError:
            return None

    df['label'] = df['breast_birads'].apply(parse_birads)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    n_b = (df['label'] == 0).sum()
    n_m = (df['label'] == 1).sum()
    print(f'[VinDr-Mammo] After BI-RADS filter: {len(df):,} rows  '
          f'Benign={n_b}  Malignant={n_m}')

    lat_col = 'laterality' if 'laterality' in df.columns else 'breast_laterality'
    view_col = 'view_position' if 'view_position' in df.columns else 'image_view'

    side_map = {'L': 'LEFT', 'R': 'RIGHT', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT'}
    df['side'] = df[lat_col].str.upper().str.strip().map(side_map)
    df['view'] = df[view_col].str.upper().str.strip()

    print(f'[VinDr-Mammo] Sides: {sorted(df["side"].dropna().unique().tolist())}')
    print(f'[VinDr-Mammo] Views: {sorted(df["view"].dropna().unique().tolist())}')

    print('[VinDr-Mammo] Indexing images ...')
    img_index: Dict[str, str] = {}
    for p in images_dir.rglob('*.png'):
        img_index[p.stem] = str(p)
    print(f'[VinDr-Mammo] Indexed {len(img_index):,} PNG files')

    id_col = 'image_id' if 'image_id' in df.columns else 'series_id'
    df['img_path'] = df[id_col].astype(str).map(img_index)
    df = df.dropna(subset=['img_path'])
    print(f'[VinDr-Mammo] Resolved paths: {len(df):,}')

    study_col = 'study_id' if 'study_id' in df.columns else 'patient_id'

    samples = []
    for study_id, grp in df.groupby(study_col):
        view_map = {}
        for _, row in grp.iterrows():
            key = (row['side'], row['view'])
            if key in REQUIRED_VIEWS:
                view_map[key] = row['img_path']
        if not REQUIRED_VIEWS.issubset(view_map.keys()):
            continue
        samples.append({
            'patient_id': str(study_id),
            'label': int(grp['label'].max()),
            'lcc': view_map[('LEFT', 'CC')],
            'lmlo': view_map[('LEFT', 'MLO')],
            'rcc': view_map[('RIGHT', 'CC')],
            'rmlo': view_map[('RIGHT', 'MLO')],
            'source': 'vindr',
        })

    print(f'[VinDr-Mammo] Complete 4-view cases: {len(samples)}')
    return samples


def build_dataloaders(
    cbis_root: Optional[str],
    vindr_root: str,
    vindr_images: Optional[str] = None,
    vindr_labels: Optional[str] = None,
    img_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation and test DataLoaders.

    Set cbis_root to None to use VinDr-Mammo only.
    """
    cbis_samples = _build_cbis_samples(cbis_root) if cbis_root else []
    vindr_samples = _build_vindr_samples(
        vindr_root, vindr_images=vindr_images, vindr_labels=vindr_labels)

    all_samples = cbis_samples + vindr_samples
    if len(all_samples) == 0:
        raise ValueError('No samples found. Check the dataset paths.')

    n_b = sum(1 for s in all_samples if s['label'] == 0)
    n_m = sum(1 for s in all_samples if s['label'] == 1)
    pw = round(n_b / max(n_m, 1), 3)
    print(f'\n  Total : {len(all_samples):,}  '
          f'Benign={n_b}  Malignant={n_m}  pos_weight={pw}\n')

    train_s, val_s, test_s = _stratified_split(all_samples, seed=seed)

    def count_source(split, src):
        return sum(1 for x in split if x.get('source') == src)

    print('-- Split --------------------------------------------------------')
    for name, split in [('Train', train_s), ('Val', val_s), ('Test', test_s)]:
        b = sum(1 for x in split if x['label'] == 0)
        m = sum(1 for x in split if x['label'] == 1)
        print(f'{name:6s}: {len(split):5d}  Benign={b}  Malignant={m}  '
              f'(CBIS={count_source(split, "cbis")} '
              f'VinDr={count_source(split, "vindr")})')
    print('-----------------------------------------------------------------\n')

    train_tfm = get_transforms(img_size, 'train')
    eval_tfm = get_transforms(img_size, 'val')
    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=(num_workers > 0))

    train_loader = DataLoader(
        FourViewDataset(train_s, train_tfm),
        batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(
        FourViewDataset(val_s, eval_tfm),
        batch_size=batch_size, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(
        FourViewDataset(test_s, eval_tfm),
        batch_size=batch_size, shuffle=False, drop_last=False, **common)

    return train_loader, val_loader, test_loader