"""Batched burial partitions are bitwise identical with and without worker processes."""
import unittest

import numpy as np

import burial_depth
import mesh_coverage
from mesh_geometry import icosphere
from parallel_runtime import RuntimePool, physical_core_count


def _stack(level=1, angle=.11):
    """A lower sheet under a rotated copy of itself: every lower face is
    covered by several upper faces, in the same pair order regions() uses."""
    lower = icosphere(level)
    upper = icosphere(level)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    upper_vertices = np.asarray(upper['vertices'])@rotation.T
    vertices = np.r_[np.asarray(lower['vertices']), upper_vertices]
    faces = np.r_[np.asarray(lower['faces']), np.asarray(upper['faces'])+len(lower['vertices'])]
    sheets = np.r_[np.zeros(len(lower['faces']), np.int64), np.ones(len(upper['faces']), np.int64)]
    overlap = mesh_coverage.material_overlaps(vertices, faces, sheets)
    first, second = overlap['first'], overlap['second']
    lower_first = sheets[first] == 0
    lower_faces = np.where(lower_first, first, second)
    upper_faces = np.where(lower_first, second, first)
    triangles = np.ascontiguousarray(vertices[faces])
    area = np.array([mesh_coverage._polygon_area(t, 6371.) for t in triangles])
    order = np.lexsort((upper_faces, lower_faces))
    unique, starts = np.unique(lower_faces[order], return_index=True)
    jobs = [(face, order[start:end], area[face], 6371.)
            for face, start, end in zip(unique, starts, np.r_[starts[1:], len(order)])]
    return triangles, upper_faces, jobs


def _same(first, second):
    (regions_a, areas_a, error_a), (regions_b, areas_b, error_b) = first, second
    def same_polygon(a,b):
        if isinstance(a,burial_depth.ExactPolygon) or isinstance(b,burial_depth.ExactPolygon):
            return type(a) is type(b) and a.homogeneous==b.homogeneous
        return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    return (len(regions_a) == len(regions_b)
            and all(same_polygon(pa,pb) and ca == cb
                    for (pa, ca), (pb, cb) in zip(regions_a, regions_b))
            and areas_a.dtype == areas_b.dtype and areas_a.tobytes() == areas_b.tobytes()
            and np.float64(error_a).tobytes() == np.float64(error_b).tobytes())


class BatchedPartitionTests(unittest.TestCase):
    def setUp(self):
        burial_depth.geometry_cache_reset()

    def tearDown(self):
        burial_depth.geometry_cache_reset()

    def serial(self, triangles, upper, jobs):
        burial_depth.geometry_cache_reset()
        session = burial_depth.partition_geometry_session(triangles, upper)
        return [session.partition_face(*job) for job in jobs]

    def test_worker_batches_match_serial_partitions_bitwise(self):
        triangles, upper, jobs = _stack()
        self.assertGreater(len(jobs), burial_depth.MIN_PARALLEL_PARTITION_FACES)
        self.assertGreater(np.mean([len(job[1]) for job in jobs]), 2.)
        expected = self.serial(triangles, upper, jobs)
        count = min(2, physical_core_count())
        burial_depth.geometry_cache_reset()
        with RuntimePool(workers=count, max_workers=count):
            session = burial_depth.partition_geometry_session(triangles, upper)
            actual = list(session.partition_faces(jobs))
            statistics = burial_depth.geometry_cache_statistics()
            # A second pass over the same geometry is served from the shared cache.
            again = list(burial_depth.partition_geometry_session(triangles, upper).partition_faces(jobs))
        self.assertEqual(len(actual), len(expected))
        self.assertTrue(all(_same(a, b) for a, b in zip(actual, expected)))
        self.assertTrue(all(_same(a, b) for a, b in zip(again, expected)))
        self.assertEqual(statistics['partition_misses'], len(jobs))
        self.assertEqual(burial_depth.geometry_cache_statistics()['partition_hits'], len(jobs))

    def test_serial_without_runtime_is_the_unchanged_per_face_path(self):
        triangles, upper, jobs = _stack()
        expected = self.serial(triangles, upper, jobs)
        burial_depth.geometry_cache_reset()
        actual = list(burial_depth.partition_geometry_session(triangles, upper).partition_faces(jobs))
        self.assertTrue(all(_same(a, b) for a, b in zip(actual, expected)))

    def test_rejected_face_raises_at_its_own_position(self):
        triangles, upper, jobs = _stack()
        # Reverse one covering triangle: the kernel rejects that face. Every
        # earlier face is still yielded, and the error is the serial error.
        bad = len(jobs)//2
        broken = triangles.copy()
        broken[upper[jobs[bad][1][0]]] = broken[upper[jobs[bad][1][0]]][::-1]
        session = burial_depth.partition_geometry_session(broken, upper)
        with self.assertRaises(ValueError) as serial_error:
            for index, job in enumerate(jobs):
                session.partition_face(*job)
        serial_index = index
        burial_depth.geometry_cache_reset()
        count = min(2, physical_core_count())
        with RuntimePool(workers=count, max_workers=count):
            batched = burial_depth.partition_geometry_session(broken, upper).partition_faces(jobs)
            seen = 0
            with self.assertRaises(ValueError) as batched_error:
                for _ in batched:
                    seen += 1
        self.assertEqual(seen, serial_index)
        self.assertEqual(str(batched_error.exception), str(serial_error.exception))


if __name__ == '__main__':
    unittest.main()
