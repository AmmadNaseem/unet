from pathlib import Path
import sys, site
root = Path(sys.prefix).resolve()
for directory in site.getsitepackages():
    packages = Path(directory)
    tensorflow = packages / 'tensorflow'
    if not tensorflow.is_dir():
        continue
    for library in packages.glob('nvidia/*/lib/*.so*'):
        target = tensorflow / library.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(library.resolve())
    for compiler in packages.glob('nvidia/*/bin/ptxas'):
        target = root / 'bin' / 'ptxas'
        if not target.exists():
            target.symlink_to(compiler.resolve())
print('Configured CUDA libraries in', root)

# The dynamic loader must receive CUDA paths before the Python process starts.
libraries = ['/usr/lib/wsl/lib']
for directory in site.getsitepackages():
    libraries.extend(str(p) for p in Path(directory).glob('nvidia/*/lib'))
kernel_path = root / 'share/jupyter/kernels/unet-gpu/kernel.json'
if kernel_path.exists():
    import json
    kernel = json.loads(kernel_path.read_text())
    kernel.setdefault('env', {})['LD_LIBRARY_PATH'] = ':'.join(libraries)
    kernel_path.write_text(json.dumps(kernel, indent=2))
    print('Configured GPU kernel:', kernel_path)
