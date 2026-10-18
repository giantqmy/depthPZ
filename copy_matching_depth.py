"""Copy regenerated depth PNGs matching RGB timestamps (without decoding)."""

import argparse
import hashlib
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-dir', type=Path, default=Path(r'I:\KEDADATA\data\fog\RGBS0'))
    parser.add_argument('--depth-dir', type=Path, default=Path(r'I:\KEDADATA\data\pz\depth'))
    parser.add_argument('--output', type=Path, default=Path(r'I:\KEDADATA\data\fog\depth_new'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--trainer-names', action='store_true',
                        help='Use timestamp_depth.png filenames for the existing trainer')
    args = parser.parse_args()
    for folder in (args.rgb_dir, args.depth_dir):
        if not folder.is_dir():
            raise FileNotFoundError(folder)
    if args.output.resolve() in (args.rgb_dir.resolve(), args.depth_dir.resolve()):
        parser.error('Output must differ from input directories')
    rgb_suffix = '_S0_rgb.png'
    depth_suffix = '_depth_dense_u16.png'
    rgb_files = sorted(args.rgb_dir.glob('*' + rgb_suffix))
    if not rgb_files:
        raise ValueError('No matching RGB files found')
    plan, missing, conflicts = [], [], []
    for rgb in rgb_files:
        prefix = rgb.name[:-len(rgb_suffix)]
        source = args.depth_dir / (prefix + depth_suffix)
        if not source.is_file():
            missing.append(rgb.name)
            continue
        name = prefix + '_depth.png' if args.trainer_names else source.name
        target = args.output / name
        if target.exists() and (not target.is_file() or digest(source) != digest(target)):
            conflicts.append(str(target))
        plan.append((source, target))
    print(f'RGB={len(rgb_files)} matched={len(plan)} missing={len(missing)} conflicts={len(conflicts)}', flush=True)
    for name in missing:
        print(f'MISSING: {name}')
    if conflicts:
        raise FileExistsError('Different destination files; nothing copied: ' + repr(conflicts))
    if args.dry_run:
        for source, target in plan[:5]:
            print(f'{source.name} -> {target.name}')
        print('Dry run: no files written')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for source, target in plan:
        if target.exists():
            skipped += 1
            continue
        # Exclusive creation protects existing files, including concurrent writes.
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if digest(source) != digest(target):
            raise IOError(f'Copy verification failed: {target}')
        copied += 1
        if copied % 100 == 0:
            print(f'Copied {copied}', flush=True)
    print(f'Done: copied={copied}, identical_skipped={skipped}, missing={len(missing)}; output={args.output}', flush=True)


if __name__ == '__main__':
    main()
