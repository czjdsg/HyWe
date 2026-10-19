"""Regressions that broke fresh-checkout workflows during the release port."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT))

from hybrid_utils import coord_scale_for_model
from collector.stage1_probe import Stage1
from collector import fact_prober_v2, refilter_pool
from evaluation.score_operation import score


def test_released_model_name_uses_training_coordinate_grid():
    assert coord_scale_for_model("HyWe-35B") == 1000
    assert coord_scale_for_model("yzxjb/HyWe-35B") == 1000
    assert coord_scale_for_model("kimi-k3") == 1


def test_probe_honors_v8_after_modules_are_already_imported(tmp_path):
    args = SimpleNamespace(sem_a=1, sem_b=1, b_conc=1, skip_existing=False,
                           max_steps=1, agent="v8")
    stage = Stage1([], tmp_path, args)
    with patch("collector.stage1_probe.run_one", new_callable=AsyncMock) as explorer:
        explorer.return_value = {"kb": {"facts": {}}}
        asyncio.run(stage.stage_a({"domain": "example.invalid"}))
        assert explorer.call_args.kwargs["agent_version"] == "v8"


def test_infrastructure_error_is_not_operation_capability_evidence():
    work = [{"domain": "example.invalid", "key": "price", "value": "42",
             "question": "What is the price?", "probe": "probe"}]
    fact_prober_v2.judge_pass(work, {}, {}, 1)
    assert work[0]["probe"] == "error"
    assert work[0]["tag"] is None


def test_legacy_error_labels_do_not_enter_refilter(tmp_path):
    rows = [{"value": "unknown", "probe": "error", "tag": "O_EXCLUSIVE"},
            {"value": "42", "probe": "probed", "tag": "S_REACHABLE"}]
    (tmp_path / "fact_tags.json").write_text(json.dumps({"facts": rows}))
    assert refilter_pool.load_tagged(tmp_path) == [rows[1]]


def test_missing_judgments_remain_in_benchmark_denominator():
    verdicts = [{"task_id": "a", "predicted_label": 1},
                {"task_id": "a", "predicted_label": None},
                {"task_id": "b", "predicted_label": None}]
    result = score(verdicts, {"a", "b", "c", "d"})
    assert result == {"total": 4, "judged": 1, "success": 1, "rate": 25.0}


def test_normalized_training_export_copies_usable_images(tmp_path):
    image = tmp_path / "rollouts/task-1/trajectory/frame.png"
    image.parent.mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (4, 4), "white").save(image)
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": str(image)}}]}]}) + "\n")
    output = tmp_path / "sft/data.jsonl"
    subprocess.run([sys.executable, str(ROOT / "training/normalize_hybrid_images.py"),
                    "--in", str(source), "--out", str(output), "--copy-images"], check=True)
    row = json.loads(output.read_text())
    portable = output.parent / row["messages"][0]["content"][0]["image_url"]["url"]
    assert portable.read_bytes() == image.read_bytes()
