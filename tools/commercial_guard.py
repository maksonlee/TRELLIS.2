"""Commercial-use guard for TRELLIS.2 generation. Import and call install() BEFORE importing trellis2 or o_voxel.

- `nvdiffrast` and `nvdiffrast.torch` become stub modules: the real package (NVIDIA Source Code License, research or
  evaluation use only) is never imported, and calling any of its functions raises immediately. Modules that merely
  import it at load time (o_voxel.postprocess, trellis2.pipelines.trellis2_texturing) still load.
- `nvdiffrec_render` (same licence) cannot be imported.
- audit() reports what was actually loaded into this process: shared objects from /proc/self/maps, real nvdiffrast /
  nvdiffrec modules, and remote-code modules of briaai/RMBG-2.0 (CC BY-NC) or ZhengPeng7/BiRefNet (MIT).
"""
import importlib.abc
import sys
import types

STUBBED = ('nvdiffrast', 'nvdiffrast.torch')
BLOCKED = ('nvdiffrec_render',)
_FUNCS = ('RasterizeCudaContext', 'RasterizeGLContext', 'rasterize', 'interpolate', 'texture', 'texture_construct_mip',
          'antialias', 'antialias_construct_topology_hash', 'DepthPeeler', 'get_log_level', 'set_log_level')


class _Disabled:
    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f'{self._name} is disabled in commercial mode '
                           '(NVIDIA Source Code License: research or evaluation use only)')


def _stub(name):
    module = types.ModuleType(name)
    module.__file__ = '<commercial_guard stub>'
    module.__path__ = []
    module.__commercial_guard_stub__ = True
    for attr in _FUNCS:
        setattr(module, attr, _Disabled(f'{name}.{attr}'))
    return module


class _BlockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise ImportError(f'{fullname} is blocked in commercial mode '
                              '(NVIDIA Source Code License: research or evaluation use only)')
        return None


def install():
    for name in STUBBED:
        current = sys.modules.get(name)
        if current is not None and not getattr(current, '__commercial_guard_stub__', False):
            raise RuntimeError(f'{name} was imported before commercial_guard.install()')
        if current is None:
            sys.modules[name] = _stub(name)
    sys.modules['nvdiffrast'].torch = sys.modules['nvdiffrast.torch']
    if not any(isinstance(finder, _BlockFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _BlockFinder())


def audit():
    shared = []
    try:
        with open('/proc/self/maps', encoding='utf-8', errors='replace') as maps:
            for line in maps:
                parts = line.split()
                path = parts[-1] if len(parts) >= 6 else ''
                if ('nvdiffrast' in path or 'nvdiffrec' in path) and path not in shared:
                    shared.append(path)
    except OSError as exc:
        shared.append(f'<cannot read /proc/self/maps: {exc}>')
    real = sorted(name for name, module in list(sys.modules.items())
                  if name.split('.')[0] in ('nvdiffrast', 'nvdiffrec_render')
                  and not getattr(module, '__commercial_guard_stub__', False))
    rmbg = sorted(name for name in list(sys.modules) if 'briaai' in name.lower() or 'rmbg' in name.lower())
    birefnet = sorted(name for name in list(sys.modules) if 'zhengpeng7' in name.lower())
    return {'nvdiffrast_or_nvdiffrec_shared_objects': shared, 'real_nvdiffrast_or_nvdiffrec_modules': real,
            'rmbg_modules': rmbg, 'birefnet_modules': birefnet,
            'clean': not shared and not real and not rmbg}
