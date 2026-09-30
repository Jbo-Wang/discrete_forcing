"""Create machine-local simulator paths; no paths are distributed with the code."""
import argparse
from pathlib import Path
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--libero-home', type=Path, default=Path('third_party/LIBERO'))
    args = parser.parse_args()
    home = args.libero_home.resolve()
    benchmark = home / 'libero/libero'
    if not (benchmark / 'bddl_files').is_dir():
        raise FileNotFoundError('Install the LIBERO repository before preparing its configuration.')
    datasets = home / 'libero/datasets'
    datasets.mkdir(exist_ok=True)
    paths = dict(benchmark_root=str(benchmark), bddl_files=str(benchmark / 'bddl_files'),
                 init_states=str(benchmark / 'init_files'), datasets=str(datasets),
                 assets=str(benchmark / 'assets'))
    target = home / 'libero/config.yaml'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    target.write_text(yaml.safe_dump(paths), encoding='utf-8')
