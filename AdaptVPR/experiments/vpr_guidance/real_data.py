"""Real benchmark adapters; ground truth always comes from declared metadata."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class RealSplit:
    references: list[Path]
    queries: list[Path]
    positives: list[list[int]]
    conditions: list[str]
    protocol: dict
    reference_poses: list | None = None
    query_poses: list | None = None

    def validate(self):
        if not self.references or not self.queries:
            raise ValueError("benchmark reference/query split is empty")
        if len(set(self.references)) != len(self.references) or len(set(self.queries)) != len(self.queries):
            raise ValueError("duplicate benchmark images")
        if len(self.queries) != len(self.positives) or len(self.queries) != len(self.conditions):
            raise ValueError("query/positives/condition length mismatch")
        for path in self.references + self.queries:
            if not path.is_file():
                raise FileNotFoundError(path)
        for positive in self.positives:
            if not positive or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                                   or not 0 <= i < len(self.references) for i in positive):
                raise ValueError("each query needs valid metadata-defined positive reference indices")
        if any(not c for c in self.conditions):
            raise ValueError("condition metadata is empty")
        return self


def image_paths(root):
    return sorted(p.resolve() for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def metadata_split(root, metadata):
    """JSON adapter for official/research splits, explicit positives or metric poses.

    pose: {center_m: [x,y,z], quaternion_wxyz: [qw,qx,qy,qz]}; world->camera
    rotation in the common COLMAP/NVM world frame. Positives are reference indices
    or exact paths, never inferred from timestamps or image basenames.
    """
    root = Path(root).resolve()
    payload = json.loads(Path(metadata).read_text())
    protocol = payload.get("protocol", {})
    if not protocol.get("source") or not protocol.get("name"):
        raise ValueError("metadata protocol must name its source and evaluation definition")
    references, queries = payload["references"], payload["queries"]
    def path(row):
        value = Path(row["path"])
        return (root / value).resolve() if not value.is_absolute() else value.resolve()
    refs = [path(r) for r in references]
    ref_map = {str(p): i for i, p in enumerate(refs)}
    ref_poses = [r.get("pose") for r in references]
    query_poses = [q.get("pose") for q in queries]
    positives = []
    centers = None
    for q in queries:
        if "positives" in q:
            items = q["positives"]
            positive = [item if isinstance(item, int) else ref_map[str(path({"path": item}))] for item in items]
        else:
            if "positive_radius_m" not in protocol or not q.get("pose") or not all(ref_poses):
                raise ValueError("need explicit positives or pose metadata with declared positive_radius_m")
            if centers is None:
                radius = float(protocol["positive_radius_m"])
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError("positive_radius_m must be finite and positive")
                centers = cKDTree(np.asarray([p["center_m"] for p in ref_poses], dtype=float))
            positive = centers.query_ball_point(q["pose"]["center_m"], radius)
        positives.append(sorted(set(positive)))
    return RealSplit(refs, [path(q) for q in queries], positives,
                     [q["condition"] for q in queries],
                     dict(protocol, metadata_path=str(Path(metadata).resolve())),
                     ref_poses, query_poses).validate()


def svox_split(root, gallery=None, query_dirs=None, radius=25.):
    """Use documented @UTM_Easting@UTM_Northing@... encoding and 25m retrieval GT."""
    root = Path(root).resolve()
    base = root / "images/test" if (root / "images/test").exists() else root
    gallery = Path(gallery) if gallery else base / "gallery"
    dirs = [Path(p) for p in query_dirs] if query_dirs else sorted(
        p for p in base.iterdir() if p.is_dir() and (p.name == "queries" or p.name.startswith("queries_")))
    refs, queries, conditions = image_paths(gallery), [], []
    def utm(path):
        parts = path.name.split("@")
        if len(parts) < 4 or parts[0] != "":
            raise ValueError(f"not documented SVOX UTM metadata: {path}; supply --metadata")
        coords = [float(parts[1]), float(parts[2])]
        if not np.isfinite(coords).all():
            raise ValueError(f"non-finite SVOX UTM: {path}")
        return coords
    if not refs or radius <= 0 or not np.isfinite(radius):
        raise ValueError("SVOX needs a nonempty gallery and positive finite radius")
    for directory in dirs:
        paths = image_paths(directory)
        queries.extend(paths)
        condition = directory.name.removeprefix("queries_") if directory.name != "queries" else "normal"
        conditions.extend([condition] * len(paths))
    tree = cKDTree(np.asarray([utm(p) for p in refs]))
    positives = tree.query_ball_point(np.asarray([utm(p) for p in queries]), radius)
    return RealSplit(refs, queries, [sorted(p) for p in positives], conditions,
                     {"name": "svox_utm_radius_retrieval", "positive_radius_m": radius,
                      "coordinate_encoding": "documented_UTM_Easting_Northing_filename_fields",
                      "source": "https://github.com/gmberton/deep-visual-geo-localization-benchmark",
                      "split": "test"}).validate()


def nordland_split(root, salad_root, reference_dir=None, query_dir=None, frame_tolerance=10):
    root = Path(root).resolve()
    metadata_root = Path(salad_root) / "datasets/Nordland"
    if reference_dir is None and query_dir is None and (root / "ref").exists() and (root / "query").exists():
        refs = [root / str(p) for p in np.load(metadata_root / "Nordland_dbImages.npy")]
        queries = [root / str(p) for p in np.load(metadata_root / "Nordland_qImages.npy")]
        positives = [list(map(int, p)) for p in np.load(metadata_root / "Nordland_gt.npy", allow_pickle=True)]
        return RealSplit(refs, queries, positives, ["winter"] * len(queries),
                         {"name": "vendored_salad_nordland", "source": str(metadata_root.resolve())}).validate()
    if frame_tolerance < 0:
        raise ValueError("frame_tolerance must be nonnegative")
    # This adapter is valid only for the locally documented prepared frame format.
    readme = root / "README.txt"
    if not readme.exists() or "10 frames" not in readme.read_text():
        raise ValueError("Nordland frame layout needs its README.txt or explicit --metadata")
    reference_dir = Path(reference_dir) if reference_dir else root / "images/test/database"
    query_dir = Path(query_dir) if query_dir else root / "images/test/queries"
    def frame(path):
        parts = path.name.split("@")
        if len(parts) <= 7 or parts[0] != "" or not parts[7].isdigit():
            raise ValueError(f"invalid documented Nordland frame metadata: {path}")
        return int(parts[7])
    refs = sorted(image_paths(reference_dir), key=frame)
    queries = sorted(image_paths(query_dir), key=frame)
    ref_ids = np.asarray([frame(p) for p in refs])
    query_ids = np.asarray([frame(p) for p in queries])
    if len(set(ref_ids)) != len(refs) or len(set(query_ids)) != len(queries):
        raise ValueError("Nordland frame IDs must be unique")
    positives = [list(range(int(np.searchsorted(ref_ids, i - frame_tolerance, side="left")),
                            int(np.searchsorted(ref_ids, i + frame_tolerance, side="right")))) for i in query_ids]
    return RealSplit(refs, queries, positives, ["winter"] * len(queries),
                     {"name": "prepared_nordland_frame_tolerance", "frame_tolerance": frame_tolerance,
                      "source": str(readme), "query_stride": 1}).validate()


def load_real_split(args):
    metadata = args.metadata or args.dataset_root / "metadata/evaluation.json"
    if metadata.exists():
        return metadata_split(args.dataset_root, metadata)
    if args.metadata:
        raise FileNotFoundError(args.metadata)
    if args.dataset == "svox":
        return svox_split(args.dataset_root, args.reference_dir, args.query_dirs, args.positive_radius)
    if args.dataset == "nordland":
        return nordland_split(args.dataset_root, args.salad_root, args.reference_dir,
                              args.query_dirs[0] if args.query_dirs else None, args.frame_tolerance)
    return robotcar_public_pose_split(args.dataset_root, args.positive_radius)



def _robotcar_image(root, name):
    """Resolve a declared official image ID, including downloaded JPEG repackaging."""
    parts = [p for p in Path(name.replace('\\', '/')).parts if p not in (".", "/", "images")]
    if parts[0] in {"left", "rear", "right"}:
        parts.insert(0, "overcast-reference")
    path = Path(root) / "images" / Path(*parts)
    if not path.exists() and path.suffix == ".png" and path.with_suffix(".jpg").exists():
        path = path.with_suffix(".jpg")
    return path.resolve()


def _colmap_pose(q, t):
    from scipy.spatial.transform import Rotation
    q = np.asarray(q, dtype=float)
    t = np.asarray(t, dtype=float)
    if not np.isfinite(q).all() or not np.isfinite(t).all() or np.linalg.norm(q) == 0:
        raise ValueError("invalid COLMAP pose")
    q /= np.linalg.norm(q)
    r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return {"center_m": (-r.T @ t).tolist(), "quaternion_wxyz": q.tolist()}


def read_colmap_reference_poses(root):
    """Read native text/binary images metadata; no 3D points or image downloads."""
    import struct
    root = Path(root).resolve()
    models = root / "3D-models"
    result = {}
    def add(name, pose):
        path = _robotcar_image(root, name)
        if "overcast-reference" not in path.parts:
            raise ValueError(f"COLMAP reference model contains non-reference image: {name}")
        if path in result and not np.allclose(result[path]["center_m"], pose["center_m"], atol=.01):
            raise ValueError("overlapping COLMAP models disagree on reference poses")
        result[path] = pose
    import io
    import re
    import zipfile

    def read_text(handle):
        while True:
            line = handle.readline()
            if not line:
                break
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 10:
                raise ValueError("invalid COLMAP image row")
            add(" ".join(fields[9:]), _colmap_pose(list(map(float, fields[1:5])), list(map(float, fields[5:8]))))
            handle.readline()  # POINTS2D row, which may be empty

    def read_binary(handle):
        count, = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            values = struct.unpack("<idddddddi", handle.read(64))
            name = bytearray()
            while True:
                char = handle.read(1)
                if not char:
                    raise ValueError("truncated COLMAP images.bin")
                if char == b"\0":
                    break
                name.extend(char)
            add(name.decode(), _colmap_pose(values[1:5], values[5:8]))
            n_points, = struct.unpack("<Q", handle.read(8))
            handle.seek(n_points * 24, 1)

    sources = set()
    files = sorted(models.rglob("images.txt"))
    files += [p for p in sorted(models.rglob("images.bin")) if not p.with_suffix(".txt").exists()]
    for path in files:
        with path.open("r" if path.suffix == ".txt" else "rb") as handle:
            (read_text if path.suffix == ".txt" else read_binary)(handle)
        sources.add(str(path))
    # Read only image pose metadata directly from the official archives.
    # This avoids extracting gigabytes of unused POINTS3D files.
    for path in sorted(models.rglob("*_aligned.zip")):
        if any(path.stem in name for name in sources):
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                names = [n for n in archive.namelist() if n.endswith("/images.txt") or n.endswith("/images.bin")]
                if len(names) != 1:
                    raise ValueError(f"expected one COLMAP images file in {path}")
                with archive.open(names[0]) as raw:
                    if names[0].endswith(".txt"):
                        with io.TextIOWrapper(raw) as handle:
                            read_text(handle)
                    else:
                        read_binary(raw)
            sources.add(str(path))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"COLMAP archive download is incomplete: {path}") from exc
    if sources and not any("all-merged" in name for name in sources):
        locations = {match.group(1) for name in sources
                     for match in [re.search(r"([0-9]{3})_aligned", name)] if match}
        if len(locations) != 49:
            raise FileNotFoundError(f"official RobotCar reference gallery needs all 49 aligned models; found {len(locations)}. "
                                    "Finish downloading the model archives before evaluating.")
    return result


def robotcar_public_pose_split(root, radius=25.):
    """Explicit v2 public-pose evaluation, distinct from its withheld official test."""
    from scipy.spatial.transform import Rotation
    from .teacher import file_sha256
    root = Path(root).resolve()
    pose_files = sorted((root / "metadata").glob("robotcar_v2_train.txt*"))
    if not pose_files:
        pose_files = sorted(root.glob("robotcar_v2_train.txt*"))
    if len(pose_files) != 1:
        raise FileNotFoundError("need exactly one official robotcar_v2_train.txt in root or metadata/")
    poses, conditions = {}, {}
    for line in pose_files[0].read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 17:
            raise ValueError("v2 public pose row must contain image_name and camera-to-world 4x4 matrix")
        transform = np.asarray(list(map(float, fields[1:]))).reshape(4, 4)
        rotation = transform[:3, :3]
        if (not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]) or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError("invalid v2 camera-to-world pose")
        q = Rotation.from_matrix(rotation.T).as_quat()
        path = _robotcar_image(root, fields[0])
        if path in poses:
            raise ValueError("duplicate v2 public pose image")
        poses[path] = {"center_m": transform[:3, 3].tolist(), "quaternion_wxyz": q[[3, 0, 1, 2]].tolist()}
        conditions[path] = fields[0].split("/")[0]
    references = {p: pose for p, pose in poses.items() if conditions[p] == "overcast-reference"}
    if not references:
        references = read_colmap_reference_poses(root)
    if not references:
        raise FileNotFoundError(
            "v2 public query poses are present, but reference poses are missing. "
            "Place official COLMAP images.txt, images.bin or *_aligned.zip under 3D-models/individual/"
            "colmap_reconstructions/<location>/. Only image pose metadata is needed; "
            "cameras/points3D files are not used by this evaluator.")
    refs = sorted(references)
    queries = sorted(p for p in poses if conditions[p] != "overcast-reference")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("RobotCar public-pose retrieval radius must be finite and positive")
    tree = cKDTree(np.asarray([references[p]["center_m"] for p in refs]))
    positives = tree.query_ball_point(np.asarray([poses[p]["center_m"] for p in queries]), radius)
    return RealSplit(refs, queries, [sorted(p) for p in positives], [conditions[p] for p in queries],
                     {"name": "robotcar_v2_public_pose_evaluation", "split": "public_poses",
                      "official_hidden_test": False, "positive_radius_m": radius,
                      "reference_condition": "overcast-reference",
                      "reference_selection": "COLMAP_registered_reference_images_with_metric_poses",
                      "source": "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/README_RobotCar_v2.md",
                      "pose_file": str(pose_files[0]), "pose_file_sha256": file_sha256(pose_files[0]),
                      "pose_convention": "world_to_camera_quaternion_and_world_camera_center"},
                     [references[p] for p in refs], [poses[p] for p in queries]).validate()
