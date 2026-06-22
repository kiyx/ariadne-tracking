"""Test per le utility pure in src/utils.py."""

import numpy as np
import torch

from src.utils import (
    aggregate_embeddings,
    get_camera_id,
    get_engine_path,
    get_video_absolute_start,
    group_by_scene,
    is_edge_bbox,
    is_partial_body,
    is_valid_roi,
    pad_and_clip_box,
    parse_video_metadata,
    recombine_tracklet_clips,
    suppress_contained_boxes,
    suppress_overlapping_boxes,
)


class TestPadAndClipBox:
    def test_padding_applied(self):
        x1, y1, x2, y2 = pad_and_clip_box(100, 100, 200, 300, 1920, 1080)
        assert x1 < 100
        assert y1 < 100
        assert x2 > 200
        assert y2 > 300

    def test_clipped_to_frame(self):
        x1, y1, _x2, _y2 = pad_and_clip_box(0, 0, 50, 100, 1920, 1080)
        assert x1 >= 0
        assert y1 >= 0


class TestIsValidRoi:
    def test_too_small(self):
        assert not is_valid_roi(10, 10)

    def test_valid(self):
        assert is_valid_roi(64, 200)

    def test_bad_aspect_ratio(self):
        assert not is_valid_roi(300, 100)


class TestSuppressContainedBoxes:
    def test_no_suppression(self):
        boxes = np.array([[0, 0, 100, 200], [200, 0, 300, 200]])
        mask = suppress_contained_boxes(boxes)
        assert mask.all()

    def test_contained_suppressed(self):
        boxes = np.array(
            [
                [0, 0, 200, 400],  # grande
                [50, 200, 150, 350],  # contenuta
            ]
        )
        mask = suppress_contained_boxes(boxes)
        assert mask[0]
        assert not mask[1]

    def test_single_box(self):
        mask = suppress_contained_boxes(np.array([[0, 0, 50, 100]]))
        assert mask[0]

    def test_empty(self):
        mask = suppress_contained_boxes(np.empty((0, 4)))
        assert len(mask) == 0

    def test_equal_area_not_mutual_suppression(self):
        """Due bbox identiche non devono sopprimersi a vicenda (tie-breaker)."""
        boxes = np.array(
            [
                [0, 0, 100, 100],  # area=10000
                [0, 0, 100, 100],  # area=10000 (identica)
            ]
        )
        mask = suppress_contained_boxes(boxes)
        assert mask[0]
        assert mask[1]

    def test_equal_area_partial_overlap_not_suppressed(self):
        """Due bbox con area uguale e sovrapposizione parziale non devono essere soppresse."""
        boxes = np.array(
            [
                [0, 0, 100, 100],  # area=10000
                [50, 50, 150, 150],  # area=10000
            ]
        )
        mask = suppress_contained_boxes(boxes)
        assert mask[0]
        assert mask[1]


class TestGetEnginePath:
    def test_convention(self):
        p = get_engine_path("models/yolo26m.pt", 640)
        assert p.name == "yolo26m_fp16_640.engine"
        assert p.parent.name == "models"


class TestGetCameraId:
    def test_standard_name(self):
        assert get_camera_id("2018-05-18.15-05-01.15-10-01.bus.G507.r13") == "G507"

    def test_unknown_format(self):
        assert get_camera_id("some_video") == "some_video"


class TestGroupByScene:
    def test_groups_same_scene(self):
        videos = [
            "2018-05-18.15-00-00.15-05-00.bus.G505.r13",
            "2018-05-18.15-05-01.15-10-01.bus.G507.r13",
        ]
        scenes = group_by_scene(videos)
        assert "2018-05-18.bus" in scenes
        assert len(scenes["2018-05-18.bus"]) == 2

    def test_different_scenes(self):
        videos = [
            "2018-05-18.15-00-00.15-05-00.bus.G505.r13",
            "2018-03-11.14-05-01.14-10-01.school.G328.r13",
        ]
        scenes = group_by_scene(videos)
        assert len(scenes) == 2


class TestSuppressOverlappingBoxes:
    def test_no_overlap(self):
        boxes = np.array([[0, 0, 100, 200], [200, 0, 300, 200]])
        mask = suppress_overlapping_boxes(boxes)
        assert mask.all()

    def test_overlap_suppresses_smaller(self):
        boxes = np.array(
            [
                [0, 0, 200, 400],  # grande (area=80000)
                [50, 50, 180, 350],  # sovrapposta (area=39000, IoU alto)
            ]
        )
        mask = suppress_overlapping_boxes(boxes)
        assert mask[0]
        assert not mask[1]

    def test_below_threshold_keeps_both(self):
        # Leggera sovrapposizione: IoU < 0.3
        boxes = np.array(
            [
                [0, 0, 100, 200],
                [80, 0, 200, 200],
            ]
        )
        mask = suppress_overlapping_boxes(boxes)
        assert mask.all()

    def test_single_box(self):
        mask = suppress_overlapping_boxes(np.array([[0, 0, 50, 100]]))
        assert mask[0]

    def test_empty(self):
        mask = suppress_overlapping_boxes(np.empty((0, 4)))
        assert len(mask) == 0

    def test_equal_area_not_mutual_suppression(self):
        """Due bbox identiche non devono sopprimersi a vicenda (tie-breaker)."""
        boxes = np.array(
            [
                [0, 0, 100, 100],  # area=10000
                [0, 0, 100, 100],  # area=10000 (identica)
            ]
        )
        mask = suppress_overlapping_boxes(boxes)
        assert mask[0]
        assert mask[1]

    def test_equal_area_partial_overlap_not_suppressed(self):
        """Due bbox con area uguale e sovrapposizione parziale non devono essere soppresse."""
        boxes = np.array(
            [
                [0, 0, 100, 100],  # area=10000
                [50, 50, 150, 150],  # area=10000
            ]
        )
        mask = suppress_overlapping_boxes(boxes)
        assert mask[0]
        assert mask[1]


class TestIsEdgeBbox:
    def test_bottom_edge_filtered(self):
        # Bbox al bordo inferiore, non al top → filtrata (piedi)
        assert is_edge_bbox(100, 800, 200, 1080, 1920, 1080)

    def test_full_body_not_filtered(self):
        # Bbox al centro del frame → non filtrata
        assert not is_edge_bbox(100, 200, 200, 600, 1920, 1080)

    def test_spans_top_to_bottom_not_filtered(self):
        # Bbox che copre tutto il frame (top + bottom) → non filtrata
        assert not is_edge_bbox(100, 0, 200, 1080, 1920, 1080)

    def test_top_only_not_filtered(self):
        # Bbox al bordo superiore ma non inferiore → non filtrata
        assert not is_edge_bbox(100, 0, 200, 400, 1920, 1080)


class TestIsPartialBody:
    """Test per il filtro ibrido pose-guided / heuristic."""

    def test_large_bbox_full_body_keypoints(self):
        # Bbox grande (h=400 >= 150), keypoint upper visibili → NON parziale
        kpt = np.zeros(17)
        kpt[0] = 0.9  # nose
        kpt[5] = 0.8  # left shoulder
        kpt[6] = 0.7  # right shoulder
        assert not is_partial_body((100, 100, 200, 500), 1920, 1080, kpt)

    def test_large_bbox_only_legs_keypoints(self):
        # Bbox grande, solo keypoint lower → parziale (piedi/gambe)
        kpt = np.zeros(17)
        kpt[13] = 0.9  # left knee
        kpt[14] = 0.8  # right knee
        kpt[15] = 0.9  # left ankle
        kpt[16] = 0.8  # right ankle
        assert is_partial_body((100, 500, 200, 900), 1920, 1080, kpt)

    def test_large_bbox_just_below_threshold_upper(self):
        # Bbox grande, solo 1 keypoint upper (sotto soglia di 2) → parziale
        kpt = np.zeros(17)
        kpt[0] = 0.9  # solo il naso
        assert is_partial_body((100, 100, 200, 500), 1920, 1080, kpt)

    def test_small_bbox_no_keypoints_at_bottom(self):
        # Bbox piccola (h=100 < 150), nessun keypoint → fallback edge → al bordo
        assert is_partial_body((100, 980, 150, 1080), 1920, 1080, None)

    def test_small_bbox_no_keypoints_center(self):
        # Bbox piccola, nessun keypoint, al centro del frame → NON parziale
        assert not is_partial_body((100, 400, 150, 500), 1920, 1080, None)

    def test_small_bbox_with_keypoints_falls_back(self):
        # Bbox piccola con keypoint → ignora i keypoint, usa edge heuristic
        kpt = np.zeros(17)  # tutti zero → verrebbe filtrata con keypoint
        # ma bbox piccola al centro → fallback edge → non al bordo → OK
        assert not is_partial_body((100, 400, 150, 500), 1920, 1080, kpt)


class TestRecombineTrackletClips:
    def test_full_segments(self):
        # 32 frame, seq_len=8, stride=4 → 32//32 = 1 segmento, 4 clip
        clips = recombine_tracklet_clips(32, 8, stride=4)
        assert len(clips) == 4
        assert all(len(c) == 8 for c in clips)

    def test_short_tracklet(self):
        # 5 frame, seq_len=8 → fallback (duplicazione)
        clips = recombine_tracklet_clips(5, 8, stride=4)
        assert len(clips) == 1
        assert len(clips[0]) == 8
        assert clips[0] == [0, 1, 2, 3, 4, 0, 1, 2]

    def test_remainder(self):
        # 36 frame, seq_len=8, stride=4 → 1 segmento (32 frame) + 4 frame rimanenti
        clips = recombine_tracklet_clips(36, 8, stride=4)
        # 4 clip dal segmento pieno + almeno una dal remainder
        assert len(clips) >= 4
        assert all(len(c) == 8 for c in clips)


class TestAggregateEmbeddings:
    def test_single_embedding(self):
        emb = torch.randn(2048)
        result = aggregate_embeddings([emb])
        assert result is not None
        assert result.shape == (2048,)
        # Deve essere L2-normalizzato
        assert torch.isclose(torch.norm(result), torch.tensor(1.0), atol=1e-5)

    def test_multiple_embeddings(self):
        embs = [torch.randn(2048) for _ in range(5)]
        result = aggregate_embeddings(embs)
        assert result is not None
        assert result.shape == (2048,)

    def test_intra_similarity_filter(self):
        # Embedding molto diversi → filtro deve scartare
        embs = [torch.randn(2048) * 10 for _ in range(5)]
        result = aggregate_embeddings(embs, min_intra_similarity=0.99)
        assert result is None

    def test_empty_list(self):
        assert aggregate_embeddings([]) is None


class TestParseVideoMetadata:
    def test_standard_name(self):
        meta = parse_video_metadata("2018-05-18.15-05-01.15-10-01.bus.G507.r13")
        assert meta is not None
        assert meta.camera_id == "G507"
        assert meta.date == "2018-05-18"
        assert meta.scene_key == "2018-05-18.bus"
        assert meta.camera_node_id == "G507_2018-05-18"

    def test_scene_pattern_fallback(self):
        meta = parse_video_metadata("2018-03-11.14-05-01.14-10-01.school.G328.r13")
        assert meta is not None
        assert meta.camera_id == "G328"
        assert meta.date == "2018-03-11"
        assert meta.scene_key == "2018-03-11.school"

    def test_unknown_format(self):
        assert parse_video_metadata("random_video") is None


class TestGetVideoAbsoluteStart:
    def test_standard_name(self):
        ts = get_video_absolute_start("2018-05-18.15-05-01.15-10-01.bus.G507.r13")
        assert ts > 0
        from datetime import datetime

        dt = datetime.fromtimestamp(ts)
        assert dt.year == 2018
        assert dt.month == 5
        assert dt.day == 18
        assert dt.hour == 15
        assert dt.minute == 5
        assert dt.second == 1

    def test_date_only_fallback(self):
        ts = get_video_absolute_start("2018-05-18.unknown.suffix")
        assert ts > 0
        from datetime import datetime

        dt = datetime.fromtimestamp(ts)
        assert dt.year == 2018
        assert dt.month == 5
        assert dt.day == 18

    def test_unparseable(self):
        assert get_video_absolute_start("random_video") == 0.0
