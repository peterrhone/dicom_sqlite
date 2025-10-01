import sqlite3
import numpy as np
import pandas as pd

class SqliteData:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(self.path)
        if not self._table_exists('dicom'):
            self.create_db()
        else:
            self._ensure_volume_and_spacing_columns()

    def _table_exists(self, table_name):
        c = self.conn.cursor()
        c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        return c.fetchone() is not None
    
    def _ensure_volume_and_spacing_columns(self):
        c = self.conn.cursor()
        c.execute("PRAGMA table_info(dicom)")
        columns = [info[1] for info in c.fetchall()]
        if 'mask_volume_ml' not in columns:
            c.execute("ALTER TABLE dicom ADD COLUMN mask_volume_ml REAL")
        if 'x_spacing' not in columns:
            c.execute("ALTER TABLE dicom ADD COLUMN x_spacing REAL")
        if 'y_spacing' not in columns:
            c.execute("ALTER TABLE dicom ADD COLUMN y_spacing REAL")
        if 'layer_spacing' not in columns:
            c.execute("ALTER TABLE dicom ADD COLUMN layer_spacing REAL")
        if 'mask_voxel_count' not in columns:
            c.execute("ALTER TABLE dicom ADD COLUMN mask_voxel_count INTEGER")
        self.conn.commit()

    def create_db(self):
        c = self.conn.cursor()
        try:
            c.execute('''
                CREATE TABLE IF NOT EXISTS dicom (
                    id INTEGER PRIMARY KEY, 
                    pat_id TEXT, 
                    cbct_id TEXT, 
                    sag_slice_idx INTEGER, 
                    cbct_image BLOB, 
                    cbct_mask BLOB, 
                    img_height INTEGER,
                    img_width INTEGER,
                    structure_id TEXT,
                    mask_volume_ml REAL,
                    mask_voxel_count INTEGER,
                    x_spacing REAL,
                    y_spacing REAL,
                    layer_spacing REAL,
                    UNIQUE(pat_id, cbct_id, sag_slice_idx, structure_id)
                )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database creation error: {e}")
            raise
    
    def _array_to_bytes(self, arr):
        if not isinstance(arr, np.ndarray):
            return None
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        return arr.tobytes()
    
    def insert_db(self, pat_id, cbct_id, sag_slice_idx, cbct_image, cbct_mask, structure_id, 
                  mask_volume_ml=None, mask_voxel_count=None, x_spacing=None, y_spacing=None, layer_spacing=None):
        c = self.conn.cursor()
        try:
            cbct_bytes = self._array_to_bytes(cbct_image.astype(np.uint16))
            mask_bytes = self._array_to_bytes(cbct_mask.astype(np.uint8)) if cbct_mask is not None else None
            c.execute('''INSERT OR IGNORE INTO dicom (
                    pat_id, cbct_id, sag_slice_idx, cbct_image, cbct_mask, 
                    img_height, img_width, structure_id, mask_volume_ml, mask_voxel_count,
                    x_spacing, y_spacing, layer_spacing) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (pat_id, cbct_id, sag_slice_idx, cbct_bytes, mask_bytes, 
                     cbct_image.shape[0], cbct_image.shape[1], structure_id, 
                     mask_volume_ml, mask_voxel_count, x_spacing, y_spacing, layer_spacing))
            self.conn.commit()
        except Exception as e:
            print(f"Error inserting data: {e}")
            raise

    def delete_by_id(self, id):
        c = self.conn.cursor()
        c.execute('''DELETE FROM dicom WHERE id=?''', (id,))
        self.conn.commit()

    def read_db(self):
        c = self.conn.cursor()
        c.execute('''SELECT * FROM dicom''')
        rows = c.fetchall()
        return rows
    
    def read_db_by_id(self, pat_id):
        c = self.conn.cursor()
        c.execute('''SELECT * FROM dicom WHERE pat_id=?''', (pat_id,))
        rows = c.fetchall()
        return rows

    def read_db_by_index(self, index):
        c = self.conn.cursor()
        c.execute('''SELECT * FROM dicom ORDER BY id LIMIT 1 OFFSET ?''', (index,))
        row = c.fetchone()
        if row is None:
            raise IndexError(f"No row found at index {index}")
        return row

    def get_total_images(self):
        c = self.conn.cursor()
        c.execute('''SELECT COUNT(*) FROM dicom''')
        return c.fetchone()[0]

    def get_images_and_masks(self, desired_mask='bladder'):
        c = self.conn.cursor()
        mask = desired_mask.lower()
        match mask:
            case 'bladder':
                query = '''SELECT cbct_image, cbct_mask, img_height, img_width, pat_id, cbct_id, 
                          mask_volume_ml, mask_voxel_count, x_spacing, y_spacing, layer_spacing 
                          FROM dicom WHERE structure_id='bladder' '''
            case 'rectum':
                query = '''SELECT cbct_image, cbct_mask, img_height, img_width, pat_id, cbct_id, 
                          mask_volume_ml, mask_voxel_count, x_spacing, y_spacing, layer_spacing 
                          FROM dicom WHERE structure_id='rectum' '''
            case 'prostate':
                query = '''SELECT cbct_image, cbct_mask, img_height, img_width, pat_id, cbct_id, 
                          mask_volume_ml, mask_voxel_count, x_spacing, y_spacing, layer_spacing 
                          FROM dicom WHERE structure_id='prostate' '''
            case _:
                raise ValueError("Invalid mask type")

        c.execute(query)
        rows = c.fetchall()
        images = []
        masks = []
        pat_ids = []
        cbct_ids = []
        volumes = []
        voxel_counts = []
        voxel_spacings = []
        for row in rows:
            img = np.frombuffer(row[0], dtype=np.uint16).reshape(row[2], row[3])
            mask = np.frombuffer(row[1], dtype=np.uint8).reshape(row[2], row[3])
            images.append(img)
            masks.append(mask)
            pat_ids.append(row[4])
            cbct_ids.append(row[5])
            volumes.append(row[6] if row[6] is not None else 0.0)
            voxel_counts.append(row[7] if row[7] is not None else 0)
            voxel_spacings.append((row[8], row[9], row[10]))
        return images, masks, pat_ids, cbct_ids, volumes, voxel_counts, voxel_spacings

    def get_midline_images_and_masks(self, desired_mask='bladder'):
        c = self.conn.cursor()
        mask = desired_mask.lower()
        
        # First, find the middle slice index for this database
        c.execute('SELECT DISTINCT sag_slice_idx FROM dicom ORDER BY sag_slice_idx')
        slice_indices = [row[0] for row in c.fetchall()]
        if not slice_indices:
            return [], [], [], []
        
        middle_idx = len(slice_indices) // 2
        midline_slice = slice_indices[middle_idx]
        
        match mask:
            case 'bladder':
                query = '''SELECT cbct_image, cbct_mask, img_height, img_width, pat_id, cbct_id 
                          FROM dicom WHERE structure_id='bladder' AND sag_slice_idx=?'''
            case 'rectum':
                query = '''SELECT cbct_image, cbct_mask, img_height, img_width, pat_id, cbct_id 
                          FROM dicom WHERE structure_id='rectum' AND sag_slice_idx=?'''
            case 'prostate':
                query = '''SELECT cbct_image, cbct_mask, img_height, img_width, pat_id, cbct_id 
                          FROM dicom WHERE structure_id='prostate' AND sag_slice_idx=?'''
            case _:
                raise ValueError("Invalid mask type")

        c.execute(query, (midline_slice,))
        rows = c.fetchall()
        images = []
        masks = []
        pat_ids = []
        cbct_ids = []
        for row in rows:
            img = np.frombuffer(row[0], dtype=np.uint16).reshape(row[2], row[3])
            mask = np.frombuffer(row[1], dtype=np.uint8).reshape(row[2], row[3])
            images.append(img)
            masks.append(mask)
            pat_ids.append(row[4])
            cbct_ids.append(row[5])
        return images, masks, pat_ids, cbct_ids

    def get_mask_volumes(self, desired_mask='bladder'):
        c = self.conn.cursor()
        mask = desired_mask.lower()
        match mask:
            case 'bladder':
                query = '''SELECT pat_id, cbct_id, structure_id, mask_volume_ml, mask_voxel_count 
                          FROM dicom WHERE structure_id='bladder' '''
            case 'rectum':
                query = '''SELECT pat_id, cbct_id, structure_id, mask_volume_ml, mask_voxel_count 
                          FROM dicom WHERE structure_id='rectum' '''
            case 'prostate':
                query = '''SELECT pat_id, cbct_id, structure_id, mask_volume_ml, mask_voxel_count 
                          FROM dicom WHERE structure_id='prostate' '''
            case _:
                raise ValueError("Invalid mask type")

        c.execute(query)
        rows = c.fetchall()
        volumes_df = pd.DataFrame(rows, columns=['pat_id', 'cbct_id', 'structure_id', 'mask_volume_ml', 'mask_voxel_count'])
        return volumes_df

    def close(self):
        self.conn.close()
