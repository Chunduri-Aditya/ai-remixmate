import importlib
import importlib.util
import tempfile
import unittest


REQUIRED_AUDIO_DEPS = ["numpy", "librosa", "scipy", "soundfile", "sklearn"]
MISSING_AUDIO_DEPS = [
    dep for dep in REQUIRED_AUDIO_DEPS
    if importlib.util.find_spec(dep) is None
]


@unittest.skipIf(
    MISSING_AUDIO_DEPS,
    f"audio dependencies not installed: {', '.join(MISSING_AUDIO_DEPS)}",
)
class AudioShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = importlib.import_module("numpy")
        cls.sf = importlib.import_module("soundfile")
        cls.dj_mixing = importlib.import_module("remixmate.dj_mixing")
        cls.remix_core = importlib.import_module("remixmate.remix_core")
        cls.timeline_planner = importlib.import_module("remixmate.timeline_planner")
        cls.timeline_renderer = importlib.import_module("remixmate.timeline_renderer")
        cls.structure_detection = importlib.import_module("remixmate.structure_detection")

    def test_apply_crossfade_keeps_stereo_shape(self):
        np = self.np
        remix_core = self.remix_core
        old_sample_rate = remix_core.SAMPLE_RATE
        remix_core.SAMPLE_RATE = 10
        try:
            stems = {
                "vocals": np.ones((20, 2), dtype=np.float32),
                "drums": np.zeros((20, 2), dtype=np.float32),
                "bass": np.zeros((20, 2), dtype=np.float32),
                "other": np.zeros((20, 2), dtype=np.float32),
            }

            result = remix_core._apply_crossfade(stems, fade_length=0.5)
        finally:
            remix_core.SAMPLE_RATE = old_sample_rate

        self.assertEqual(result.shape, (20, 2))
        self.assertTrue(np.all(result[-1] < result[0]))
        self.assertTrue(np.allclose(result[-1], 0.0))

    def test_align_stems_trims_and_fills_none_as_stereo(self):
        np = self.np
        remix_core = self.remix_core
        stems = {
            "vocals": np.ones((8, 2), dtype=np.float32),
            "drums": np.ones(6, dtype=np.float32) * 2.0,
            "bass": None,
            "other": np.ones((10, 2), dtype=np.float32) * 4.0,
        }

        aligned = remix_core._align_stems_to_same_length(stems)

        self.assertEqual(set(aligned), {"vocals", "drums", "bass", "other"})
        for stem in aligned.values():
            self.assertEqual(stem.shape, (6, 2))

        self.assertTrue(np.allclose(aligned["drums"][:, 0], 2.0))
        self.assertTrue(np.allclose(aligned["drums"][:, 1], 2.0))
        self.assertTrue(np.allclose(aligned["bass"], 0.0))

    def test_time_stretch_handles_mono_and_stereo(self):
        np = self.np
        samples = np.sin(np.linspace(0, 16 * np.pi, 4096, dtype=np.float32))
        stereo = np.column_stack([samples, samples * 0.5])

        stretched_stereo = self.dj_mixing.time_stretch_audio(stereo, rate=2.0, sr=22050)
        stretched_mono = self.dj_mixing.time_stretch_audio(samples, rate=2.0, sr=22050)

        self.assertEqual(stretched_stereo.ndim, 2)
        self.assertEqual(stretched_stereo.shape[1], 2)
        self.assertLess(stretched_stereo.shape[0], stereo.shape[0])
        self.assertEqual(stretched_mono.ndim, 1)
        self.assertLess(stretched_mono.shape[0], samples.shape[0])

    def test_apply_eq_filter_operates_on_sample_axis(self):
        np = self.np
        t = np.linspace(0, 1, 512, endpoint=False, dtype=np.float32)
        stereo = np.column_stack([
            np.sin(2 * np.pi * 220 * t),
            np.cos(2 * np.pi * 220 * t),
        ])

        filtered = self.dj_mixing.apply_eq_filter(stereo, low_cut=0.2, sr=44100)

        self.assertEqual(filtered.shape, stereo.shape)
        self.assertFalse(np.allclose(filtered, stereo))

    def test_timeline_renderer_accepts_stereo_stems_and_outputs_stereo(self):
        np = self.np
        Clip = self.timeline_planner.Clip
        ArrangementPlan = self.timeline_planner.ArrangementPlan
        TimelineRenderer = self.timeline_renderer.TimelineRenderer

        plan = ArrangementPlan(
            clips=[
                Clip(
                    source_track="A",
                    stem="vocals",
                    start_time_src=0.0,
                    end_time_src=1.0,
                    start_time_out=0.0,
                    fade_in=0.2,
                    fade_out=0.2,
                )
            ],
            total_duration=1.0,
        )
        stems_a = {
            "vocals": np.column_stack([
                np.ones(10, dtype=np.float32),
                np.ones(10, dtype=np.float32) * 0.5,
            ])
        }
        stems_b = {}

        rendered = TimelineRenderer(sr=10).render_timeline(
            plan, stems_a, stems_b, bpm_a=120.0, bpm_b=120.0, target_bpm=120.0
        )

        self.assertEqual(rendered.shape, (10, 2))
        self.assertGreater(np.max(np.abs(rendered)), 0)
        self.assertTrue(np.all(rendered[:, 0] >= rendered[:, 1]))

    def test_beatmatching_helper_stretches_each_present_stem(self):
        np = self.np
        remix_core = self.remix_core
        original_time_stretch = remix_core.time_stretch_audio
        calls = []

        def fake_time_stretch(stem, rate, sr):
            calls.append((stem.shape, rate, sr))
            return stem[:4]

        remix_core.time_stretch_audio = fake_time_stretch
        try:
            stems = {
                "vocals": np.ones((8, 2), dtype=np.float32),
                "drums": np.ones((8, 2), dtype=np.float32),
                "bass": None,
            }
            processed = remix_core._apply_beatmatching_to_stems(
                stems,
                target_bpm=120.0,
                source_bpm=100.0,
                max_stretch=1.25,
            )
        finally:
            remix_core.time_stretch_audio = original_time_stretch

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(abs(call[1] - 1.2) < 1e-9 for call in calls))
        self.assertEqual(processed["vocals"].shape, (4, 2))
        self.assertIsNone(processed["bass"])

    def test_harmonic_shift_selects_small_compatible_shift(self):
        remix_core = self.remix_core

        self.assertEqual(remix_core._harmonic_shift_semitones("8A", "8A"), 0)
        self.assertEqual(remix_core._harmonic_shift_semitones("8A", "11A"), -2)

    def test_timeline_planner_uses_energy_shape_and_aggressiveness(self):
        Section = self.structure_detection.Section
        TimelinePlanner = self.timeline_planner.TimelinePlanner

        sections_a = [
            Section("A", "intro", 0.0, 4.0, 0.2, False, 4),
            Section("A", "build", 4.0, 8.0, 0.5, False, 4),
            Section("A", "drop", 8.0, 20.0, 0.9, True, 8),
            Section("A", "outro", 20.0, 24.0, 0.2, False, 4),
        ]
        sections_b = [
            Section("B", "intro", 0.0, 4.0, 0.2, False, 4),
            Section("B", "drop", 4.0, 24.0, 0.7, True, 8),
            Section("B", "drop", 24.0, 68.0, 0.95, True, 16),
            Section("B", "outro", 68.0, 72.0, 0.2, False, 4),
        ]

        planner = TimelinePlanner()
        conservative = planner.plan_arrangement(
            sections_a,
            sections_b,
            beats_a=[],
            beats_b=[],
            genre="edm",
            mode="mashup",
            aggressiveness=0.0,
            energy_shape="slow_build",
        )
        aggressive = planner.plan_arrangement(
            sections_a,
            sections_b,
            beats_a=[],
            beats_b=[],
            genre="edm",
            mode="mashup",
            aggressiveness=1.0,
            energy_shape="rollercoaster",
        )

        self.assertIsNotNone(conservative.energy_curve)
        self.assertIsNotNone(aggressive.energy_curve)
        self.assertNotEqual(conservative.energy_curve, aggressive.energy_curve)
        self.assertIn("aggressiveness=0.0", conservative.explanation)
        self.assertIn("aggressiveness=1.0", aggressive.explanation)
        self.assertGreater(aggressive.total_duration, conservative.total_duration)

    def test_stem_status_helper_preserves_legacy_dict_return(self):
        remix_core = self.remix_core
        old_demucs_available = remix_core.DEMUCS_AVAILABLE
        audio_path = "/tmp/nonexistent-fallback-test.wav"

        try:
            remix_core.DEMUCS_AVAILABLE = False
            status = remix_core.separate_stems_with_status(audio_path)
            legacy_paths = remix_core.separate_stems(audio_path)
        finally:
            remix_core.DEMUCS_AVAILABLE = old_demucs_available

        self.assertEqual(status["method"], "fallback")
        self.assertFalse(status["success"])
        self.assertEqual(status["source_audio_path"], audio_path)
        self.assertIn("Demucs is not installed", status["warning"])
        self.assertEqual(legacy_paths, status["stem_paths"])
        self.assertEqual(
            set(legacy_paths),
            {"vocals", "drums", "bass", "other"},
        )

    def test_metadata_path_and_stem_status_formatting(self):
        remix_core = self.remix_core

        self.assertTrue(
            remix_core.metadata_path_for_output("/tmp/remix.wav").endswith(
                "remix_metadata.json"
            )
        )
        self.assertTrue(
            remix_core.metadata_path_for_output("/tmp/custom.wav").endswith(
                "custom_metadata.json"
            )
        )

        demucs_message = remix_core.format_stem_status({
            "stem_separation": {
                "track_a": {"method": "demucs", "success": True},
                "track_b": {"method": "demucs", "success": True},
            }
        })
        self.assertEqual(demucs_message, "True Demucs stems used for both tracks.")

        fallback_message = remix_core.format_stem_status({
            "stem_separation": {
                "track_a": {
                    "method": "fallback",
                    "success": False,
                    "warning": "Demucs is not installed.",
                },
                "track_b": {
                    "method": "fallback",
                    "success": False,
                    "warning": "Demucs is not installed.",
                },
            }
        })
        self.assertIn("Fallback mode used", fallback_message)
        self.assertIn("full-track audio as each stem", fallback_message)

    def test_fallback_remix_writes_stereo_output_and_metadata(self):
        np = self.np
        sf = self.sf
        remix_core = self.remix_core
        old_demucs_available = remix_core.DEMUCS_AVAILABLE
        sr = 44100

        with tempfile.TemporaryDirectory() as tmp_dir:
            t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
            track_a = f"{tmp_dir}/track_a.wav"
            track_b = f"{tmp_dir}/track_b.wav"
            output_path = f"{tmp_dir}/classic.wav"

            audio_a = 0.2 * np.sin(2 * np.pi * 220 * t)
            audio_b = 0.2 * np.sin(2 * np.pi * 330 * t)
            sf.write(track_a, np.column_stack([audio_a, audio_a * 0.8]), sr)
            sf.write(track_b, np.column_stack([audio_b * 0.8, audio_b]), sr)

            try:
                remix_core.DEMUCS_AVAILABLE = False
                result_path = remix_core.remix_two_files(
                    track_a,
                    track_b,
                    mode="mashup",
                    apply_beatmatching=False,
                    output_path=output_path,
                )
            finally:
                remix_core.DEMUCS_AVAILABLE = old_demucs_available

            rendered, output_sr = sf.read(result_path, always_2d=True)
            metadata = remix_core.load_remix_metadata(result_path)

        self.assertEqual(output_sr, sr)
        self.assertEqual(rendered.ndim, 2)
        self.assertEqual(rendered.shape[1], 2)
        self.assertGreater(np.max(np.abs(rendered)), 0)

        for track_key in ("track_a", "track_b"):
            track_status = metadata["stem_separation"][track_key]
            self.assertEqual(track_status["method"], "fallback")
            self.assertFalse(track_status["success"])
            self.assertIn("Demucs is not installed", track_status["warning"])
            self.assertEqual(
                set(track_status["stem_paths"]),
                {"vocals", "drums", "bass", "other"},
            )


if __name__ == "__main__":
    unittest.main()
