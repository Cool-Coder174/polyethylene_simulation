

import pandas as pd
from io import StringIO

def fix_pdb_file(pdb_path):
    """
    Reads a PDB file, adds missing terminal hydrogens, renumbers atoms,
    and rewrites the file in a compliant PDB format.

    This function uses pandas.read_fwf to robustly parse ATOM records,
    avoiding errors with misaligned columns.
    """
    with open(pdb_path, 'r') as f:
        pdb_content = f.read()

    # Separate ATOM, CONECT, and other lines
    lines = pdb_content.strip().split('\n')
    atom_lines = [line for line in lines if line.startswith('ATOM')]
    other_lines = [line for line in lines if not (line.startswith('ATOM') or line.startswith('CONECT') or line.startswith('END'))]

    # Use pandas to robustly parse ATOM lines, handling potential whitespace issues
    colspecs = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26), (26, 27), 
                (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78)]
    names = ['record', 'serial', 'name', 'altLoc', 'resName', 'chainID', 'resSeq', 'iCode', 
             'x', 'y', 'z', 'occupancy', 'tempFactor', 'element']
    
    atom_df = pd.read_fwf(StringIO("\n".join(atom_lines)), colspecs=colspecs, header=None, names=names)
    atom_df['altLoc'] = atom_df['altLoc'].fillna('')
    atom_df['iCode'] = atom_df['iCode'].fillna('')
    atom_df['chainID'] = atom_df['chainID'].fillna('A')
    atom_df['resName'] = atom_df['resName'].fillna('PE')
    atom_df['element'] = atom_df['element'].fillna('')
    atom_df['name'] = atom_df['name'].str.strip()

    # Add missing hydrogens to the terminal carbons
    # H13 for C1
    c1_coords = atom_df.loc[atom_df['name'] == 'C1'][['x', 'y', 'z']].iloc[0]
    h13_coords = c1_coords + pd.Series({'x': -1.04, 'y': 0.0, 'z': 0.0})
    h13_row = {'record': 'ATOM', 'name': ' H13', 'resName': 'PE', 'resSeq': 1, 'chainID': 'A',
               'x': h13_coords.x, 'y': h13_coords.y, 'z': h13_coords.z,
               'occupancy': 1.0, 'tempFactor': 0.0, 'element': 'H', 'altLoc': '', 'iCode': ''}
    
    # H103 for C10
    c10_coords = atom_df.loc[atom_df['name'].str.startswith('C10')][['x', 'y', 'z']].iloc[0]
    h103_coords = c10_coords + pd.Series({'x': 1.04, 'y': 0.0, 'z': 0.0})
    h103_row = {'record': 'ATOM', 'name': 'H103', 'resName': 'PE', 'resSeq': 1, 'chainID': 'A',
                'x': h103_coords.x, 'y': h103_coords.y, 'z': h103_coords.z,
                'occupancy': 1.0, 'tempFactor': 0.0, 'element': 'H', 'altLoc': '', 'iCode': ''}

    # Append new rows and re-index
    atom_df = pd.concat([atom_df, pd.DataFrame([h13_row, h103_row])], ignore_index=True)
    atom_df['serial'] = range(1, len(atom_df) + 1)

    # Fill any missing residue sequence numbers; crucial for preventing NaN conversion errors.
    atom_df['resSeq'] = atom_df['resSeq'].fillna(1)

    # Ensure occupancy and tempFactor are numeric, coercing errors to defaults.
    atom_df['occupancy'] = pd.to_numeric(atom_df['occupancy'], errors='coerce').fillna(1.0)
    atom_df['tempFactor'] = pd.to_numeric(atom_df['tempFactor'], errors='coerce').fillna(0.0)

    # Format and write back to PDB
    new_atom_lines = []
    for _, row in atom_df.iterrows():
        # This formatting string ensures compliance with PDB format standards.
        new_line = (f"{str(row['record']):<6}{int(row['serial']):>5} {str(row['name']):<4}{str(row['altLoc']):<1}{str(row['resName']):<3} {str(row['chainID']):<1}"
                    f"{int(row['resSeq']):>4}{str(row['iCode']):<1}   {row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}"
                    f"{row['occupancy']:6.2f}{row['tempFactor']:6.2f}          {str(row['element']):>2}")
        new_atom_lines.append(new_line)

    # OpenMM can infer bonds, so CONECT records are not essential for this workflow.
    final_pdb_content = "\n".join(other_lines + new_atom_lines) + "\nEND\n"

    with open(pdb_path, 'w') as f:
        f.write(final_pdb_content)

    print(f"SUCCESS: PDB file '{pdb_path}' has been cleaned and updated.")

if __name__ == "__main__":
    pdb_file_path = '/home/i.hernandez-domingu/Github/polyethylene_simulation/data/polyethylene_chain.pdb'
    fix_pdb_file(pdb_file_path)


