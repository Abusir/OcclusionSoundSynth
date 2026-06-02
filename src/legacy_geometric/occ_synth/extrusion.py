from __future__ import annotations

from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import triangulate

from .scene_generator import Scene2D
from soundspaces_adapter.material_database import scene_material_assignment


MATERIALS = {
    "floor": (0.55, 0.55, 0.50),
    "ceiling": (0.78, 0.80, 0.82),
    "outdoor_ceiling": (0.35, 0.55, 0.80),
    "sky_absorber": (0.35, 0.55, 0.80),
    "wall": (0.70, 0.72, 0.74),
    "obstacle": (0.35, 0.34, 0.31),
    "indoor_floor_hard": (0.55, 0.55, 0.50),
    "outdoor_ground": (0.44, 0.50, 0.38),
    "outdoor_ground_grass": (0.30, 0.48, 0.22),
    "outdoor_ground_soil": (0.42, 0.34, 0.24),
    "indoor_wall_reflective": (0.70, 0.72, 0.74),
    "indoor_ceiling_reflective": (0.78, 0.80, 0.82),
    "solid_occluder": (0.35, 0.34, 0.31),
}


def _coord_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), 8), round(float(y), 8))


def _ring_coords(poly: Polygon) -> list[tuple[float, float]]:
    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(float(x), float(y)) for x, y in coords]


def _write_materials(path: Path) -> None:
    lines: list[str] = []
    for name, rgb in MATERIALS.items():
        lines += [
            f"newmtl {name}",
            f"Kd {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}",
            "Ka 0.050 0.050 0.050",
            "Ks 0.080 0.080 0.080",
            "Ns 8.0",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


class ObjWriter:
    def __init__(self, obj_path: Path, material_file_name: str) -> None:
        self.obj_path = obj_path
        self.lines: list[str] = [
            f"mtllib {material_file_name}",
            "o occ_synth_scene",
        ]
        self.vertex_count = 0

    def add_vertex(self, x: float, y: float, z: float) -> int:
        self.lines.append(f"v {x:.6f} {z:.6f} {y:.6f}")
        self.vertex_count += 1
        return self.vertex_count

    def add_face(self, indices: list[int], material: str, group: str) -> None:
        if len(indices) < 3:
            return
        self.lines.append(f"g {group}")
        self.lines.append(f"usemtl {material}")
        self.lines.append("f " + " ".join(str(i) for i in indices))

    def add_triangle_faces(self, triangles: list[list[int]], material: str, group: str) -> None:
        if not triangles:
            return
        self.lines.append(f"g {group}")
        self.lines.append(f"usemtl {material}")
        for indices in triangles:
            if len(indices) == 3:
                self.lines.append("f " + " ".join(str(i) for i in indices))

    def save(self) -> None:
        self.obj_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _add_extruded_polygon(
    writer: ObjWriter,
    polygon: Polygon,
    height: float,
    side_material: str,
    group_prefix: str,
    floor_material: str = "floor",
    ceiling_material: str = "ceiling",
) -> None:
    coords = _ring_coords(polygon)
    bottom = [writer.add_vertex(x, y, 0.0) for x, y in coords]
    top = [writer.add_vertex(x, y, height) for x, y in coords]
    coord_to_bottom = {_coord_key(*coords[i]): bottom[i] for i in range(len(coords))}
    coord_to_top = {_coord_key(*coords[i]): top[i] for i in range(len(coords))}

    floor_tris: list[list[int]] = []
    ceiling_tris: list[list[int]] = []
    for tri in triangulate(polygon):
        if not polygon.covers(tri.representative_point()):
            continue
        tri_coords = _ring_coords(tri)
        floor_tris.append([coord_to_bottom[_coord_key(x, y)] for x, y in reversed(tri_coords)])
        ceiling_tris.append([coord_to_top[_coord_key(x, y)] for x, y in tri_coords])

    writer.add_triangle_faces(floor_tris, floor_material, f"{group_prefix}_floor")
    writer.add_triangle_faces(ceiling_tris, ceiling_material, f"{group_prefix}_ceiling")

    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        writer.add_triangle_faces(
            [
                [bottom[i], bottom[j], top[j]],
                [bottom[i], top[j], top[i]],
            ],
            side_material,
            f"{group_prefix}_side",
        )


def _add_floor_polygon(
    writer: ObjWriter,
    polygon: Polygon,
    group_prefix: str,
    floor_material: str = "floor",
) -> None:
    coords = _ring_coords(polygon)
    vertices = [writer.add_vertex(x, y, 0.0) for x, y in coords]
    coord_to_vertex = {_coord_key(*coords[i]): vertices[i] for i in range(len(coords))}

    floor_tris: list[list[int]] = []
    for tri in triangulate(polygon):
        if not polygon.covers(tri.representative_point()):
            continue
        tri_coords = _ring_coords(tri)
        floor_tris.append([coord_to_vertex[_coord_key(x, y)] for x, y in reversed(tri_coords)])

    writer.add_triangle_faces(floor_tris, floor_material, f"{group_prefix}_floor")


def export_scene_obj(scene: Scene2D, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / f"{scene.scene_id}.obj"
    mtl_path = output_dir / f"{scene.scene_id}.mtl"
    _write_materials(mtl_path)
    writer = ObjWriter(obj_path, mtl_path.name)

    material_assignment = scene_material_assignment(scene.scene_type)
    floor_mat = material_assignment["floor"] or "floor"
    boundary_mat = material_assignment["open_boundary" if scene.is_outdoor else "wall"] or "wall"
    ceiling_mat = material_assignment["open_ceiling" if scene.is_outdoor else "ceiling"] or "ceiling"
    obstacle_assignment = material_assignment["obstacle"]
    obstacle_mat = obstacle_assignment or "solid_occluder"
    if scene.is_outdoor:
        _add_extruded_polygon(
            writer,
            scene.boundary,
            scene.height_m,
            boundary_mat,
            "boundary",
            floor_material=floor_mat,
            ceiling_material=ceiling_mat,
        )
        writer.lines.append("# outdoor scene: finite domain with strongly absorbing side and top boundaries")
    for idx, obstacle in enumerate(scene.obstacles):
        obstacle_height = scene.height_m if not scene.is_outdoor else min(scene.height_m, 3.5)
        _add_extruded_polygon(
            writer,
            obstacle,
            obstacle_height,
            obstacle_mat,
            f"obstacle_{idx:02d}",
            floor_material=obstacle_mat,
            ceiling_material=obstacle_mat,
        )
    if not scene.is_outdoor:
        _add_extruded_polygon(
            writer,
            scene.boundary,
            scene.height_m,
            boundary_mat,
            "boundary",
            floor_material=floor_mat,
            ceiling_material=ceiling_mat,
        )

    # Keep a searchable material marker for validators and downstream tools.
    writer.lines.append(f"# semantic_floor_material {floor_mat}")
    writer.lines.append(f"# semantic_boundary_material {boundary_mat}")
    writer.lines.append(f"# semantic_ceiling_material {ceiling_mat}")
    writer.lines.append(f"# semantic_obstacle_material {obstacle_assignment or 'none'}")
    writer.save()
    return {"obj": str(obj_path), "mtl": str(mtl_path)}
