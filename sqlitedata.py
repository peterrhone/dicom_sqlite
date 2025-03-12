import sqlite3
import numpy as np

class SqliteData:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(self.path)
        if not self._table_exists('dicom'):
            self.create_db()
    
    def _table_exists(self, table_name):
        # check if table exists in database
        c = self.conn.cursor()
        c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        return c.fetchone() is not None
    
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
                    UNIQUE(pat_id, cbct_id, sag_slice_idx, structure_id)
                )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database creation error: {e}")
            raise
    
    # def insert_db(self, pat_id, cbct_num, sag_slice_num, cbct_image, cbct_bladder_mask, cbct_rectum_mask):
    #     c = self.conn.cursor()
    #     c.execute('''INSERT INTO dicom (pat_id, cbct_id, sag_slice_num, cbct_image, cbct_bladder_mask, cbct_rectum_mask) VALUES (?, ?, ?, ?, ?, ?)''', \
    #               (pat_id, cbct_num, sag_slice_num, cbct_image, cbct_bladder_mask, cbct_rectum_mask))
    #     self.conn.commit()

    def _array_to_bytes(self, arr):
        """Convert numpy array to bytes, ensuring C-contiguous"""
        # check first ir arr is really an array
        if not isinstance(arr, np.ndarray):
            return None
            # raise ValueError("Input must be a numpy array")
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        return arr.tobytes()
    
    def insert_db(self, pat_id, cbct_id, sag_slice_idx, cbct_image, cbct_mask, structure_id): 
        c = self.conn.cursor()
        try:
            # Convert arrays to bytes
            cbct_bytes = self._array_to_bytes(cbct_image.astype(np.uint16))
            mask_bytes = self._array_to_bytes(cbct_mask) if cbct_mask is not None else None
            
            c.execute('''INSERT OR IGNORE INTO dicom (
                      pat_id, 
                      cbct_id, 
                      sag_slice_idx, 
                      cbct_image, 
                      cbct_mask, 
                      img_height, 
                      img_width,
                      structure_id) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (pat_id, cbct_id, sag_slice_idx, cbct_bytes, 
                      mask_bytes, cbct_image.shape[0], cbct_image.shape[1], structure_id))
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

    def get_total_images(self):
        c = self.conn.cursor()
        c.execute('''SELECT COUNT(*) FROM dicom''')
        return c.fetchone()[0]

    def read_db_by_index(self, index):
        c = self.conn.cursor()
        c.execute('''SELECT * FROM dicom LIMIT 1 OFFSET ?''', (index,))
        return c.fetchone()

    def get_images_and_masks(self, desired_mask='bladder'):
        c = self.conn.cursor()
        mask = desired_mask.lower()
        match mask:
            case 'bladder':
                c.execute('''SELECT cbct_image, cbct_mask, img_height, img_width, pat_id FROM dicom WHERE structure_id='bladder' ''')
            case 'rectum':
                c.execute('''SELECT cbct_image, cbct_mask, img_height, img_width, pat_id FROM dicom WHERE structure_id='rectum' ''')
            case 'prostate':
                c.execute('''SELECT cbct_image, cbct_mask, img_height, img_width, pat_id FROM dicom WHERE structure_id='prostate' ''')
            case _:
                raise ValueError("Invalid mask type")

        rows = c.fetchall()
        images = []
        masks = []
        pat_ids = []
        for row in rows:
            img = np.frombuffer(row[0], dtype=np.uint16).reshape(row[2], row[3])
            mask = np.frombuffer(row[1], dtype=np.int32).reshape(row[2], row[3])
            images.append(img)
            masks.append(mask)
            pat_ids.append(row[4])
        return images, masks, pat_ids

    def close(self):
        self.conn.close()