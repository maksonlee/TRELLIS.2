# Generate GLB files for commercial use

`generate_commercial.py` turns one image into textured GLB files with the same steps and default settings as the web
app's **Generate** and **Extract GLB** (`app.py`), without the 48-view preview. It replaces the components whose licences
do not allow commercial use:

| Step | Upstream default | This tool |
|---|---|---|
| Background removal | `briaai/RMBG-2.0` (CC BY-NC 4.0) | `ZhengPeng7/BiRefNet` (MIT), pinned to revision `e2bf8e44` |
| UV rasterization for texture baking (`o_voxel.postprocess.to_glb`) | nvdiffrast (NVIDIA Source Code License: research or evaluation only) | `cubvh_raster.py` on CuMesh's cuBVH (MIT) |
| Preview rendering | nvdiffrast and nvdiffrec | none; open the GLB in Blender or another viewer |
| Texture format | WebP (`EXT_texture_webp`) | PNG (default) or WebP |

`commercial_guard.py` runs first in the process:

- It replaces `nvdiffrast` with a stub whose functions raise, so the real package is never imported.
- It blocks `nvdiffrec_render`.
- At exit it records which shared objects and modules were loaded.

The result JSON contains this audit under `commercial_audit`. A run whose audit is not clean exits with code 3.

## Usage

Set up the environment as described in the main README; the native extensions built by `setup.sh` are required. Then
run from the repository root:

```bash
mkdir -p out
MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 \
python tools/generate_commercial.py \
  --input path/to/image.png \
  --output out/model-1536-seed{seed}.glb \
  --resolution 1536 --seeds 42 \
  --decimation 1000000 --texture 4096 \
  --save-latent out/model-1536-seed{seed}-latent.npz \
  --result out/model-1536.json
```

| Option | Default | Meaning |
|---|---|---|
| `--resolution` | `1024` | `512`, `1024` (`1024_cascade`) or `1536` (`1536_cascade`) |
| `--seeds` | `42` | one seed or a list such as `42,7,123`; seeds share one model load, and the output name needs `{seed}` |
| `--decimation` | `500000` | target face count (the web app allows 100,000–1,000,000) |
| `--texture` | `2048` | texture size (the web app allows 1024–4096) |
| `--texture-format` | `png` | `png` or `webp` |
| `--save-latent` | none | also keep the result before export (`.npz`), see below; with several seeds the name needs `{seed}` |
| `--from-latent` | none | export a kept result instead of generating; replaces `--input`, `--resolution` and `--seeds` |
| `--result` | none | JSON with timings, memory, outputs and the licence audit |

- Exit codes: `0` success; `2` some seeds ran out of GPU memory (each seed fails on its own); `3` licence audit not clean;
  `1` other errors.
- An existing output file is never overwritten.
- The two `MALLOC_` variables keep glibc from holding on to freed CPU copies of the model weights. Without them, the
  web app was killed for running out of memory on its second generation (32 GB, no swap).

## Export again with other settings

The face count and texture size are export settings. `to_glb` simplifies the generated mesh to the target, unwraps new
UVs, and bakes the textures from the generated 3D data. `--save-latent` keeps the result before export: the shape and
texture latents that the web app keeps between **Generate** and **Extract GLB**. At 1536 the file is about 3.4 MB.

`--from-latent` exports such a file again with other settings. It loads only the two decoders, so it needs no image,
background removal or DINOv3:

```bash
MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 \
python tools/generate_commercial.py \
  --from-latent out/model-1536-seed42-latent.npz \
  --output out/model-1536-seed42-100k.glb \
  --decimation 100000 --texture 4096 \
  --result out/model-1536-100k.json
```

- The file holds the shape and texture features, their coordinates and the resolution, plus JSON metadata: seed,
  sampler settings, input image SHA-256, background-removal model and model revision. It is read without pickle.
- It decodes to the same model only with the same decoders. If the cached `microsoft/TRELLIS.2-4B` revision differs
  from the one in the file, the export still runs and the result JSON records the difference.
- Decimating a finished GLB in another tool is not the same. That mesh is split along its UV seams, and the old
  textures are reused instead of being baked again.

## Licences of the generation path

This is my reading of the licence texts, not legal advice. Check the current terms before you ship anything.

| Component | Licence |
|---|---|
| TRELLIS.2 code and weights (`microsoft/TRELLIS.2-4B`, `microsoft/TRELLIS-image-large`) | MIT |
| DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) | DINOv3 License: royalty-free use, including commercial use, with conditions (trade controls, prohibited uses such as military applications, redistribution under the agreement, indemnity); gated on Hugging Face |
| BiRefNet (`ZhengPeng7/BiRefNet`) | MIT |
| CuMesh (including cubvh and xatlas), FlexGEMM, o-voxel, utils3d | MIT |
| FlashAttention | BSD-3-Clause |

- PyTorch, transformers and the other Python packages and CUDA libraries were not reviewed.
- `app.py`, `example.py` and the other upstream scripts are unchanged. They still use RMBG-2.0, nvdiffrast and nvdiffrec.

## Verification

Tested on Windows 11 with WSL 2 Ubuntu 26.04 and an RTX 5060 Ti 16GB, using Python 3.10, PyTorch 2.8.0+cu128, CUDA 12.8,
and upstream commit `75fbf01`.

`test_bake_equivalence.py` compares `cubvh_raster.py` with nvdiffrast. Using nvdiffrast here is an evaluation use. The
test runs the chunk loop of `to_glb` on the UV layouts of existing GLB files:

| Mesh | Texture | Texels covered by both | Only one of them | Same triangle | Position difference (mean / max, model size ≈ 1) |
|---|---|---|---|---|---|
| 947,735 faces | 4096 | 8,715,518 | 8,637 / 10,067 | 99.14% | 6.0e-8 / 6.2e-4 |
| 496,782 faces | 2048 | 2,197,738 | 2,697 / 3,205 | 98.74% | 1.5e-7 / 2.8e-4 |

- The texels that differ lie on triangle edges. `to_glb` inpaints uncovered texels afterwards.
- A full `to_glb` export with `cubvh_raster.py` took 11.7 s, against 11.8–12.4 s with nvdiffrast (500,000 faces, 2048
  px). Its remeshing step is not deterministic: two nvdiffrast runs gave 489,550 and 490,168 faces. Exported textures
  therefore cannot be compared pixel by pixel.
- A 1536 run with 1,000,000 faces and 4096 px PNG textures took 4 min 18 s, and its audit was clean. Blender 5.2
  imported the GLB with the base-colour and metallic-roughness textures connected.
- With `--save-latent`, a 1536 run (seed 42) wrote a 3,516,178-byte latent. Exports from it decoded to the same raw
  mesh as the generation, 4,025,468 vertices and 8,081,906 faces.
  - At 100,000 faces and 4096 px, an export took 57 s with a peak of 4.3 GiB process memory. Generating again took 236 s
    and 19.5 GiB.
  - Four 100,000-face GLBs of the same generation had 99,002 to 99,622 faces. Compared with the first, the mean
    surface distance of each was under 0.01% of the bounding-box diagonal.
