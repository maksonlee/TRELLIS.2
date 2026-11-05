"""nvdiffrast-free UV rasterization for o_voxel.postprocess.to_glb, built on CuMesh's cuBVH (MIT, from ashawkey/cubvh).

to_glb uses three nvdiffrast calls to bake textures: RasterizeCudaContext(), rasterize(ctx, pos, tri, resolution) on
2D triangles in UV space (pos = [1, V, 4] with z = 0 and w = 1, called in chunks of 100k faces), and
interpolate(attr, rast, tri). This module provides those three with the same output layout:

  rast[..., 0:2]  barycentric weights of the triangle's vertices 0 and 1 (vertex 2 gets 1 - b0 - b1)
  rast[..., 3]    triangle index + 1 within `tri` (0 = no triangle)
  row r, column c samples the texel centre at NDC ((c + 0.5) / W * 2 - 1, (r + 0.5) / H * 2 - 1)

For every texel centre, cuBVH returns the nearest triangle and the barycentric weights of the nearest point. A texel is
covered when that distance is (numerically) zero, i.e. its centre lies inside the triangle. Only this UV baking case is
supported; anything else raises instead of guessing.
"""
import torch
from cumesh import cuBVH


class RasterizeCudaContext:
    def __init__(self, device=None):
        self.device = device


def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    if ranges is not None or pos.dim() != 3 or pos.shape[0] != 1 or pos.shape[2] != 4:
        raise ValueError('cubvh_raster.rasterize supports one UV-space minibatch [1, V, 4] without ranges')
    p = pos[0].float()
    if p[:, 2].abs().max().item() != 0 or (p[:, 3] - 1).abs().max().item() != 0:
        raise ValueError('cubvh_raster.rasterize supports 2D UV-space triangles only (z = 0, w = 1)')
    height, width = int(resolution[0]), int(resolution[1])
    device = p.device
    rast = torch.zeros((1, height, width, 4), device=device, dtype=torch.float32)
    tri = tri.long()
    xy = p[:, :2].contiguous()
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    keep = torch.nonzero(area2.abs() > 1e-12).squeeze(1)  # degenerate UV triangles cover no texel centre
    if keep.numel() == 0:
        return rast, None
    faces = tri[keep]
    if faces.shape[0] < 9:  # cuBVH requires more than 8 triangles; repeats map back to the same triangle
        faces = faces.repeat(-(-9 // faces.shape[0]), 1)
    bvh = cuBVH(torch.cat([xy, torch.zeros_like(xy[:, :1])], dim=1), faces.int())
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height * 2 - 1
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    points = torch.stack([gx, gy, torch.zeros_like(gx)], dim=-1).reshape(-1, 3)
    dist, face_id, uvw = bvh.unsigned_distance(points, return_uvw=True)
    inside = torch.nonzero(dist <= 1e-3 * 2.0 / max(height, width)).squeeze(1)  # within 1/1000 of a texel
    out = rast.view(-1, 4)
    out[inside, 0] = uvw[inside, 0]
    out[inside, 1] = uvw[inside, 1]
    out[inside, 3] = (keep[face_id[inside] % keep.numel()] + 1).float()
    return rast, None


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    if attr.dim() != 3 or attr.shape[0] != 1 or rast_db is not None or diff_attrs is not None:
        raise ValueError('cubvh_raster.interpolate supports attr [1, V, C] without derivatives')
    height, width = rast.shape[1], rast.shape[2]
    r = rast[0].reshape(-1, 4)
    ids = r[:, 3].round().long()
    covered = ids > 0
    out = torch.zeros((height * width, attr.shape[2]), device=attr.device, dtype=attr.dtype)
    f = tri.long()[ids[covered] - 1]
    b0, b1 = r[covered, 0:1].to(attr.dtype), r[covered, 1:2].to(attr.dtype)
    values = attr[0]
    out[covered] = values[f[:, 0]] * b0 + values[f[:, 1]] * b1 + values[f[:, 2]] * (1 - b0 - b1)
    return out.view(1, height, width, -1), None
