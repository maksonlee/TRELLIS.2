"""Generate textured GLBs from one image with TRELLIS.2 using only components whose licences allow commercial use.

  python generate_commercial.py --input IMG --output OUT.glb [--resolution 1536|1024|512] [--seeds 42[,7,...]]
                                [--decimation 500000] [--texture 2048] [--texture-format png|webp]
                                [--result RESULT.json] [--preprocessed PRE.png]

Same steps and settings as the web app's Generate + Extract GLB (app.py defaults, without its 48-view preview), except:
  * background removal: ZhengPeng7/BiRefNet (MIT) at a fixed revision instead of briaai/RMBG-2.0 (CC BY-NC); the
    pipeline config still names RMBG-2.0, but TRELLIS.2's BiRefNet wrapper class is replaced before loading;
  * texture baking: the UV rasterization in o_voxel.postprocess.to_glb uses cubvh_raster (CuMesh cuBVH, MIT) instead
    of nvdiffrast (NVIDIA Source Code License, research/evaluation only);
  * commercial_guard stubs nvdiffrast and blocks nvdiffrec_render for the whole process, and the result records an
    audit of what was loaded (shared objects and modules); a non-clean audit fails the run (exit 3);
  * textures are written as PNG by default (wider importer support than EXT_texture_webp).
Run from the TRELLIS.2 repository root. Several seeds share one model load; with more than one seed, OUT must contain
"{seed}". An existing output file is never overwritten (a time suffix is added). A GPU out-of-memory error fails only
that seed. 1536_cascade lowers the resolution by itself (steps of 128, not below 1024) above 49152 tokens; the resolution
actually used is recorded.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commercial_guard  # noqa: E402

commercial_guard.install()  # before anything can import nvdiffrast
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path.cwd().resolve()
if not ((REPO / "app.py").is_file() and (REPO / "trellis2").is_dir()):
    sys.exit("generate_commercial.py must be run from the TRELLIS.2 repository root")
if str(REPO) not in sys.path:
    sys.path.insert(1, str(REPO))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from transformers import AutoModelForImageSegmentation  # noqa: E402
import o_voxel  # noqa: E402
from trellis2.pipelines import Trellis2ImageTo3DPipeline as Pipeline  # noqa: E402
from trellis2.pipelines import rembg as rembg_module  # noqa: E402
import cubvh_raster  # noqa: E402

BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
REQUESTED_REMBG = []


class PinnedBiRefNet(rembg_module.BiRefNet):
    """TRELLIS.2's BiRefNet wrapper, always loading the MIT-licensed ZhengPeng7/BiRefNet at a fixed revision."""

    def __init__(self, model_name=None):
        REQUESTED_REMBG.append(model_name)
        self.model = AutoModelForImageSegmentation.from_pretrained(BIREFNET_REPO, revision=BIREFNET_REVISION,
                                                                   trust_remote_code=True)
        self.model.eval()
        self.transform_image = transforms.Compose([  # identical to TRELLIS.2's wrapper
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


rembg_module.BiRefNet = PinnedBiRefNet
o_voxel.postprocess.dr = cubvh_raster  # to_glb's RasterizeCudaContext / rasterize / interpolate

# app.py's UI defaults (Advanced Settings)
SS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
SHAPE = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
TEX = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}
SKIP_1024 = {"shape_slat_flow_model_1024", "tex_slat_flow_model_1024"}
T0 = time.monotonic()
result = {"started_utc": datetime.now(timezone.utc).isoformat(), "stages": [], "status": "RUNNING",
          "sampler_params": {"sparse_structure": SS, "shape_slat": SHAPE, "tex_slat": TEX},
          "malloc": {k: os.environ.get(k) for k in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_ARENA_MAX")},
          "commercial": {"background_removal": f"{BIREFNET_REPO}@{BIREFNET_REVISION} (MIT)",
                         "uv_rasterizer": "cubvh_raster.py on CuMesh cuBVH (MIT)",
                         "guard": "nvdiffrast stubbed, nvdiffrec_render blocked"}}
RESULT_PATH = None


def meminfo_mib(key):
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) // 1024
    return -1


def stage(name, **extra):
    entry = {"stage": name, "t_s": round(time.monotonic() - T0, 1),
             "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
             "mem_available_mib": meminfo_mib("MemAvailable"),
             "cuda_peak_alloc_mib": torch.cuda.max_memory_allocated() // 2**20,
             "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20, **extra}
    result["stages"].append(entry)
    print(f"[gen] {entry['t_s']:7.1f}s  {name}", flush=True)
    if RESULT_PATH:
        RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main(a):
    skip = SKIP_1024 if a.resolution == "512" else set()  # 1024_cascade and 1536_cascade need the 1024 flow models
    Pipeline.model_names_to_load = [n for n in Pipeline.model_names_to_load if n not in skip]
    pipeline = Pipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = True
    pipeline.cuda()
    if not isinstance(pipeline.rembg_model, PinnedBiRefNet):
        raise RuntimeError("background-removal model was not replaced; refusing to continue")
    result["loaded_models"] = sorted(pipeline.models)
    result["commercial"]["rembg_requested_by_pipeline_config"] = REQUESTED_REMBG
    result["commercial"]["rembg_loaded"] = getattr(pipeline.rembg_model.model.config, "_name_or_path", None)
    stage(f"models loaded ({len(pipeline.models)})")

    with Image.open(a.input) as im:
        im.load()
        result["input_image"] = {"path": a.input, "size": list(im.size), "mode": im.mode}
        image = pipeline.preprocess_image(im)
    result["preprocessed_image"] = {"size": list(image.size), "mode": image.mode}
    if a.preprocessed:
        image.save(a.preprocessed)
    stage("input prepared (background removed, cropped)")

    pipeline_type = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[a.resolution]
    seeds = [int(s) for s in a.seeds.split(",")]
    result["outputs"] = []
    for seed in seeds:
        entry = {"seed": seed, "pipeline_type": pipeline_type}
        result["outputs"].append(entry)
        out = Path(a.output.replace("{seed}", str(seed)))
        if out.exists():  # never overwrite
            out = out.with_name(f"{out.stem}-{datetime.now():%Y%m%d-%H%M%S}{out.suffix}")
        outputs = latents = shape_slat = _tex_slat = mesh = asset = None
        oom = False
        try:
            torch.cuda.reset_peak_memory_stats()
            t = time.monotonic()
            outputs, latents = pipeline.run(image, seed=seed, preprocess_image=False,
                                            sparse_structure_sampler_params=SS, shape_slat_sampler_params=SHAPE,
                                            tex_slat_sampler_params=TEX, pipeline_type=pipeline_type,
                                            return_latent=True)
            shape_slat, _tex_slat, res = latents
            mesh = outputs[0]
            entry.update({"generate_seconds": round(time.monotonic() - t, 1), "output_resolution": res,
                          "shape_slat_tokens": int(shape_slat.coords.shape[0]),
                          "raw_mesh": {"vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0])}})
            outputs = latents = shape_slat = _tex_slat = None
            torch.cuda.empty_cache()
            stage(f"seed {seed}: 3D generated ({pipeline_type}, resolution {res}, {entry['generate_seconds']} s)",
                  seed=seed, cuda_phase="generate")

            torch.cuda.reset_peak_memory_stats()
            t = time.monotonic()
            asset = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                attr_layout=pipeline.pbr_attr_layout, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=a.decimation, texture_size=a.texture, remesh=True, remesh_band=1,
                remesh_project=0, use_tqdm=False)
            mesh = None
            tmp = out.with_name(out.stem + ".partial.glb")
            asset.export(str(tmp), extension_webp=(a.texture_format == "webp"))
            asset = None
            os.replace(tmp, out)
            data = out.read_bytes()
            entry.update({"status": "PASS", "path": str(out), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest(), "decimation_target": a.decimation,
                          "texture_size": a.texture, "texture_format": a.texture_format,
                          "export_seconds": round(time.monotonic() - t, 1)})
            stage(f"seed {seed}: GLB exported ({len(data):,} bytes, {entry['export_seconds']} s)",
                  seed=seed, cuda_phase="export")
        except Exception as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not oom:
                raise
            # A GPU out-of-memory error fails only this seed; the next seed still runs.
            entry.update({"status": "CUDA_OOM", "error": str(exc)[:600]})
        if oom:
            outputs = latents = shape_slat = _tex_slat = mesh = asset = None
            gc.collect()
            torch.cuda.empty_cache()
            stage(f"seed {seed}: FAILED - GPU out of memory", seed=seed)
    failed = [e["seed"] for e in result["outputs"] if e.get("status") != "PASS"]
    result["failed_seeds"] = failed
    result["status"] = "PASS" if not failed else "CUDA_OOM"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resolution", choices=("1536", "1024", "512"), default="1024")
    ap.add_argument("--seeds", default="42", help="one seed or a comma-separated list, e.g. 42,7,123")
    ap.add_argument("--decimation", type=int, default=500000)
    ap.add_argument("--texture", type=int, default=2048)
    ap.add_argument("--texture-format", choices=("png", "webp"), default="png")
    ap.add_argument("--result")
    ap.add_argument("--preprocessed")
    args = ap.parse_args()
    seed_list = args.seeds.split(",")
    if not all(s.strip().isdigit() for s in seed_list):
        sys.exit("--seeds must be whole numbers, e.g. 42 or 42,7,123")
    if len(seed_list) > 1 and "{seed}" not in args.output:
        sys.exit("--output must contain {seed} when several seeds are given")
    RESULT_PATH = Path(args.result) if args.result else None
    result["args"] = vars(args)
    try:
        main(args)
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "CUDA_OOM"
        result["error"] = str(exc)[:2000]
        print("[gen] FAILED: GPU out of memory:", str(exc)[:400], flush=True)
    except Exception as exc:  # recorded, not hidden
        result["status"] = "CUDA_OOM" if "out of memory" in str(exc).lower() else "ERROR"
        result["error"] = "".join(traceback.format_exception(exc))[-4000:]
        print("[gen] FAILED:", result["error"], flush=True)
    finally:
        result["commercial_audit"] = commercial_guard.audit()
        if not result["commercial_audit"]["clean"]:
            result["status"] = "AUDIT_FAILED"
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["elapsed_total_s"] = round(time.monotonic() - T0, 1)
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        if RESULT_PATH:
            RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[gen] commercial audit: {'clean' if result['commercial_audit']['clean'] else 'NOT CLEAN'}", flush=True)
        print(f"[gen] result: {result['status']}", flush=True)
    sys.exit({"PASS": 0, "CUDA_OOM": 2, "AUDIT_FAILED": 3}.get(result["status"], 1))
