"""Evaluation: does cubvh_raster (CuMesh cuBVH, MIT) bake the same textures as nvdiffrast in o_voxel's to_glb?

Uses the real nvdiffrast only to compare results (evaluation use). Produces no deliverable model.
  A. Rasterization stage only, on UV meshes from existing GLBs: the exact chunk loop of to_glb, then interpolation of
     vertex positions; coverage and position differences.
  B. End to end on a freshly generated 512 mesh (BiRefNet background removal): to_glb with nvdiffrast (A1), with
     cubvh_raster (B), and with nvdiffrast again (A2, run-to-run baseline); geometry and texture differences.

  python test_bake_equivalence.py --out DIR --image IMG GLB[:TEXSIZE] [GLB[:TEXSIZE] ...]
Run from the TRELLIS.2 repository root.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.getcwd())

import numpy as np
import torch
import trimesh
from PIL import Image
import nvdiffrast.torch as real_dr  # evaluation use only
import cubvh_raster


def bake_positions(dr, vertices, faces, uvs, size):
    """Lines 229-249 of o_voxel/postprocess.py, verbatim apart from names."""
    ctx = dr.RasterizeCudaContext()
    uvs_rast = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, size, size, 4), device='cuda', dtype=torch.float32)
    for i in range(0, faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(ctx, uvs_rast, faces[i:i + 100000], resolution=[size, size])
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)
    mask = rast[0, ..., 3] > 0
    pos = dr.interpolate(vertices.unsqueeze(0), rast, faces)[0][0]
    return mask, pos, rast[0, ..., 3].round().long()


def timed(fn):
    torch.cuda.synchronize()
    t = time.monotonic()
    out = fn()
    torch.cuda.synchronize()
    return out, round(time.monotonic() - t, 2)


def part_a(glb, size):
    scene = trimesh.load(glb, force='scene', process=False)
    (geom,) = scene.geometry.values()
    v = torch.tensor(np.asarray(geom.vertices), dtype=torch.float32, device='cuda')
    f = torch.tensor(np.asarray(geom.faces), dtype=torch.int32, device='cuda')
    uv = torch.tensor(np.asarray(geom.visual.uv), dtype=torch.float32, device='cuda')  # = to_glb's out_uvs
    (ma, pa, ia), ta = timed(lambda: bake_positions(real_dr, v, f, uv, size))
    (mb, pb, ib), tb = timed(lambda: bake_positions(cubvh_raster, v, f, uv, size))
    both = ma & mb
    d = (pa - pb).abs().amax(-1)[both]
    extent = float((v.max(0).values - v.min(0).values).max())
    return {
        'glb': os.path.basename(glb), 'faces': int(f.shape[0]), 'texture_size': size,
        'seconds': {'nvdiffrast': ta, 'cubvh_raster': tb},
        'covered': {'nvdiffrast': int(ma.sum()), 'cubvh_raster': int(mb.sum()), 'both': int(both.sum()),
                    'only_nvdiffrast': int((ma & ~mb).sum()), 'only_cubvh_raster': int((mb & ~ma).sum())},
        'same_triangle_fraction': round(float((ia[both] == ib[both]).float().mean()), 6),
        'position_abs_diff': {'max': float(d.max()), 'mean': float(d.mean()),
                              'fraction_above_1e-5': round(float((d > 1e-5).float().mean()), 8)},
        'model_extent': round(extent, 4),
    }


def texture_arrays(asset):
    mat = asset.visual.material
    return (np.asarray(mat.baseColorTexture.convert('RGBA'), dtype=np.int16),
            np.asarray(mat.metallicRoughnessTexture.convert('RGB'), dtype=np.int16))


def compare_assets(x, y):
    same_geometry = (x.vertices.shape == y.vertices.shape and x.faces.shape == y.faces.shape
                     and np.array_equal(np.asarray(x.faces), np.asarray(y.faces))
                     and np.allclose(np.asarray(x.vertices), np.asarray(y.vertices), atol=1e-6)
                     and np.allclose(np.asarray(x.visual.uv), np.asarray(y.visual.uv), atol=1e-6))
    out = {'same_geometry_and_uv': bool(same_geometry),
           'faces': [int(x.faces.shape[0]), int(y.faces.shape[0])]}
    if same_geometry:
        for name, a, b in zip(('base_color_rgba', 'metallic_roughness'), texture_arrays(x), texture_arrays(y)):
            diff = np.abs(a - b)
            mse = float((diff.astype(np.float64) ** 2).mean())
            out[name] = {'mean_abs': round(float(diff.mean()), 4), 'max_abs': int(diff.max()),
                         'pixels_any_channel_diff_gt_2': round(float((diff.max(-1) > 2).mean()), 6),
                         'psnr_db': round(10 * np.log10(255 ** 2 / mse), 2) if mse > 0 else 'identical'}
    return out


def part_b(image_path, out_dir):
    import o_voxel
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    from trellis2.pipelines import Trellis2ImageTo3DPipeline as Pipeline
    from trellis2.pipelines import rembg as rembg_module

    class PinnedBiRefNet(rembg_module.BiRefNet):  # same replacement as generate_commercial.py
        def __init__(self, model_name=None):
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4", trust_remote_code=True)
            self.model.eval()
            self.transform_image = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                                            [0.229, 0.224, 0.225])])

    rembg_module.BiRefNet = PinnedBiRefNet
    Pipeline.model_names_to_load = [n for n in Pipeline.model_names_to_load
                                    if n not in ("shape_slat_flow_model_1024", "tex_slat_flow_model_1024")]
    pipeline = Pipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = True
    pipeline.cuda()
    with Image.open(image_path) as im:
        image = pipeline.preprocess_image(im.convert('RGB'))
    ss = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    shape = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    tex = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}
    outputs, latents = pipeline.run(image, seed=42, preprocess_image=False, sparse_structure_sampler_params=ss,
                                    shape_slat_sampler_params=shape, tex_slat_sampler_params=tex,
                                    pipeline_type="512", return_latent=True)
    mesh, res = outputs[0], latents[2]
    del outputs, latents
    torch.cuda.empty_cache()

    def export(dr):
        o_voxel.postprocess.dr = dr
        return o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=500000, texture_size=2048, remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False)

    a1, t_a1 = timed(lambda: export(real_dr))
    b, t_b = timed(lambda: export(cubvh_raster))
    a2, t_a2 = timed(lambda: export(real_dr))
    for tag, asset in (('nvdiffrast', a1), ('cubvh_raster', b)):
        asset.visual.material.baseColorTexture.convert('RGB').resize((1024, 1024)).save(
            os.path.join(out_dir, f'b-basecolor-{tag}.png'))
    if compare_assets(a1, b)['same_geometry_and_uv']:
        diff = np.abs(texture_arrays(a1)[0][..., :3] - texture_arrays(b)[0][..., :3]).max(-1)
        Image.fromarray(np.clip(diff * 16, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, 'b-basecolor-diff-x16.png'))
    return {'resolution': int(res), 'seconds': {'nvdiffrast_1': t_a1, 'cubvh_raster': t_b, 'nvdiffrast_2': t_a2},
            'nvdiffrast_1_vs_cubvh_raster': compare_assets(a1, b),
            'nvdiffrast_1_vs_nvdiffrast_2': compare_assets(a1, a2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('glbs', nargs='+')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {'torch': torch.__version__, 'part_a': [], 'part_b': None}
    for spec in a.glbs:
        path, _, size = spec.rpartition(':') if spec.rsplit(':', 1)[-1].isdigit() else (spec, '', '4096')
        report['part_a'].append(part_a(path, int(size)))
        print(json.dumps(report['part_a'][-1]), flush=True)
        torch.cuda.empty_cache()
    report['part_b'] = part_b(a.image, a.out)
    print(json.dumps(report['part_b']), flush=True)
    with open(os.path.join(a.out, 'equivalence.json'), 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
