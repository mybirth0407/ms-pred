"""Merge a NIST'23 structure-only SDF export + its spectra MSP into a combined SDF.

Some NIST'23 exports ship structures (.sdf: molblocks only, no spectral fields) and
spectra (.msp: peaks + adduct + collision energy + NIST# + InChIKey, no structure) as
SEPARATE files. ms-pred's reformat_nist_lcmsms_sdf.py needs a COMBINED SDF where each
molblock carries the spectral `> <FIELD>` properties. This script produces that.

The two files are only ~positionally aligned (local indels) and their NAME strings
disagree (SDF encodes Greek letters as raw high bytes like 'Γ‘'; MSP uses
'.beta.'/'.alpha.'), so positional and name matching both desync. Instead we join on
the RDKit InChIKey, order-independently:
  1. From BOTH structure SDFs, build full-InChIKey -> molblock (exact stereo) and
     ikey14 -> molblock (connectivity fallback). Consecutive duplicate molblocks reuse
     the cached key, so RDKit runs ~once per distinct structure, not per spectrum.
  2. Stream every MSP spectrum; bind it to the structure whose full InChIKey matches
     the spectrum's InChIKey field (falling back to the 14-char connectivity key), and
     emit molblock + <FIELD> property blocks. Every spectrum with a known structure is
     kept regardless of ordering/indels; exact-stereo binding reproduces the upstream
     (inchikey, adduct, instrument) grouping used by the shipped splits.

Then run reformat_nist_lcmsms_sdf.py on the output to get labels.tsv + spec_files.hdf5.
Empirically reproduces ~99.4% of the shipped nist23 split IDs.

Example:
    python data_scripts/nist23/build_combined_sdf.py \\
        --raw-dir /path/to/nist23_export --out /tmp/combined_nist23.sdf
    python <ms-data-parser>/reformat_nist_lcmsms_sdf.py \\
        --input-file /tmp/combined_nist23.sdf \\
        --targ-dir data/spec_datasets/nist23 --dataset nist2023 --workers 32
"""
import argparse
from itertools import groupby

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

KEY_MAPPING = {
    'Name': 'NAME', 'Notes': 'NOTES', 'Precursor_type': 'PRECURSOR TYPE',
    'Spectrum_type': 'SPECTRUM TYPE', 'PrecursorMZ': 'PRECURSOR M/Z',
    'Instrument_type': 'INSTRUMENT TYPE', 'Instrument': 'INSTRUMENT',
    'Sample_inlet': 'SAMPLE INLET', 'Ionization': 'IONIZATION',
    'Collision_gas': 'COLLISION GAS', 'Collision_energy': 'COLLISION ENERGY',
    'Ion_mode': 'ION MODE', 'InChIKey': 'INCHIKEY', 'Synon': 'SYNONYMS',
    'Formula': 'FORMULA', 'MW': 'MW', 'ExactMass': 'EXACT MASS', 'CAS#': 'CASNO',
    'Related_CAS#': 'RELATED CASNO', 'NIST#': 'NISTNO', 'DB#': 'ID',
    'Comments': 'COMMENT', 'Num Peaks': 'NUM PEAKS',
    'In-source_voltage': 'IN-SOURCE VOLTAGE', 'msN_pathway': 'MSN PATHWAY',
    'Peptide_sequence': 'PEPTIDE SEQUENCE', 'Peptide_mods': 'PEPTIDE MODS',
    'Retention_index': 'RETENTION INDEX', 'COMPOUND_REP': 'COMPOUND REP',
    'Salt': 'SALT', 'Salt/mix_CAS#': 'SALT/MIX CASNO', 'Known_impurity': 'KNOWN IMPURITY',
}
# Default NIST'23 export file stems (structure SDF + spectra MSP), paired by index.
DEFAULT_STEMS = ['hr_msms_nist_complete', 'hr_msms_nist_complete_2']


def sdf_blocks(path, encoding):
    with open(path, 'r', encoding=encoding, errors='replace') as fp:
        for is_delim, lines in groupby(fp, key=lambda x: "$$" in x):
            if is_delim:
                continue
            yield list(lines)


def msp_blocks(path, encoding):
    with open(path, 'r', encoding=encoding, errors='replace') as fp:
        for is_blank, lines in groupby(fp, key=lambda x: x == "\n"):
            if is_blank:
                continue
            block = list(lines)
            if block and block[0].startswith("Name:"):
                yield block


def molblock_inchikey(mb):
    """Return (full_inchikey, ikey14) computed by RDKit, or (None, None)."""
    m = Chem.MolFromMolBlock(mb)
    if m is None:
        return None, None
    try:
        smi = Chem.MolToSmiles(m, isomericSmiles=True)
        m2 = Chem.MolFromSmiles(smi)
        if m2 is None:
            return None, None
        full = Chem.MolToInchiKey(m2)
        return full, full[:14]
    except Exception:
        return None, None


def build_structure_map(sdf_files, encoding, limit):
    """Build full-InChIKey -> molblock (exact stereo) and ikey14 -> molblock (fallback)."""
    full_to_mol = {}
    conn_to_mol = {}
    n = 0
    prev_text = None
    prev_full = prev_conn = None
    for path in sdf_files:
        for sdf_lines in sdf_blocks(path, encoding):
            if 'M  END' not in sdf_lines[-1]:
                sdf_lines.append('M  END\n')
            text = "".join(sdf_lines)
            if text == prev_text:            # consecutive duplicate structure: reuse keys
                full, conn = prev_full, prev_conn
            else:
                full, conn = molblock_inchikey(text)
                prev_text, prev_full, prev_conn = text, full, conn
            if full is not None and full not in full_to_mol:
                full_to_mol[full] = sdf_lines
            if conn is not None and conn not in conn_to_mol:
                conn_to_mol[conn] = sdf_lines
            n += 1
            if limit and n >= limit:
                break
        if limit and n >= limit:
            break
    print(f'  structure map: {n} SDF blocks -> {len(full_to_mol)} full-ikey, '
          f'{len(conn_to_mol)} ikey14', flush=True)
    return full_to_mol, conn_to_mol


def msp_inchikey(block):
    for l in block:
        if l.startswith("InChIKey:"):
            return l.split(":", 1)[1].strip()
    return None


def msp_to_property_block(block):
    meta_dict = {}
    for is_meta, info_lines in groupby(block, key=lambda x: ':' in x):
        if is_meta:
            for line in info_lines:
                if '#' in line and len(line.split(':')) > 2:
                    entries = line.strip().split(';')
                else:
                    entries = [line.strip()]
                for l in entries:
                    if ':' not in l:
                        continue
                    key, val = l.split(':', 1)
                    mapped = KEY_MAPPING.get(key.strip())
                    if mapped is None:
                        continue
                    meta_dict.setdefault(mapped, []).append(f'{val.strip()}\n')
        else:
            peaks = []
            for line in info_lines:
                sp = line.split()
                if len(sp) >= 2:
                    peaks.append(f'{sp[0]} {sp[1]}\n')
            meta_dict['MASS SPECTRAL PEAKS'] = peaks
    return meta_dict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', required=True,
                    help='directory holding the NIST23 export (structure .sdf + spectra .msp)')
    ap.add_argument('--stems', nargs='+', default=DEFAULT_STEMS,
                    help='paired <stem>.sdf/<stem>.msp file stems (order matters)')
    ap.add_argument('--out', required=True, help='output combined SDF path')
    ap.add_argument('--encoding', default='iso-8859-7')
    ap.add_argument('--limit', type=int, default=0, help='cap SDF blocks scanned (testing)')
    ap.add_argument('--msp-limit', type=int, default=0, help='cap MSP spectra emitted (testing)')
    args = ap.parse_args()

    sdf_files = [f'{args.raw_dir}/{s}.sdf' for s in args.stems]
    msp_files = [f'{args.raw_dir}/{s}.msp' for s in args.stems]

    print('Building structure map from SDF files...', flush=True)
    full_to_mol, conn_to_mol = build_structure_map(sdf_files, args.encoding, args.limit)

    written = dropped_nostruct = dropped_noikey = 0
    used_full = used_conn = 0
    with open(args.out, 'w', encoding='utf-8') as out_fp:
        for path in msp_files:
            print(f'Streaming spectra from {path}', flush=True)
            for block in msp_blocks(path, args.encoding):
                ik = msp_inchikey(block)
                if not ik:
                    dropped_noikey += 1
                    continue
                mol = full_to_mol.get(ik)          # exact stereo structure
                if mol is not None:
                    used_full += 1
                else:
                    mol = conn_to_mol.get(ik[:14])  # connectivity fallback
                    if mol is not None:
                        used_conn += 1
                if mol is None:
                    dropped_nostruct += 1
                    continue
                meta = msp_to_property_block(block)
                append_lines = [f'> <{kk}>\n' + ''.join(v) + '\n' for kk, v in meta.items()]
                append_lines.append('$$$$\n')
                out_fp.writelines(mol)
                out_fp.writelines(append_lines)
                written += 1
                if args.msp_limit and written >= args.msp_limit:
                    break
            if args.msp_limit and written >= args.msp_limit:
                break
    print(f'DONE. written={written} (exact_stereo={used_full}, conn_fallback={used_conn}) '
          f'dropped_no_structure={dropped_nostruct} dropped_no_ikey={dropped_noikey}', flush=True)


if __name__ == '__main__':
    main()
