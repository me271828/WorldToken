from __future__ import annotations

from diffusion_wm.eval_rollout import episode_meta_row_fields, summarize_episodes


def _row(task: str, success: bool, layout_id: int, style_id: int, category: str) -> dict:
    meta = {
        "layout_id": layout_id,
        "style_id": style_id,
        "object_cfgs": [
            {
                "name": "obj",
                "obj_groups": category,
                "info": {
                    "cat": category,
                    "mjcf_path": f"objects/{category}/model.xml",
                },
            }
        ],
        "fixtures": {"coffee_machine": {"cls": "CoffeeMachine"}},
        "fixture_refs": {"coffee_machine": "coffee_machine"},
        "lang": "dummy",
    }
    return {"task": task, "success": success, **episode_meta_row_fields(task, meta)}


def test_episode_meta_fields_and_rollout_summary_groups() -> None:
    rows = [
        _row("CoffeeSetupMug", True, 6, 9, "mug"),
        _row("CoffeeSetupMug", False, 7, 10, "mug"),
        _row("PnPCabToCounter", True, 1, 1, "bowl"),
    ]

    summary = summarize_episodes(rows)

    assert summary["per_task_family"]["coffee"]["episodes"] == 2
    assert summary["per_task_family"]["pick_place"]["success_rate"] == 1.0
    assert summary["per_bc_eval_scene_group"]["bc_eval_style_9_10"]["episodes"] == 2
    assert summary["per_bc_eval_scene_group"]["bc_eval_style_9_10"]["success_rate"] == 0.5
    assert summary["per_style"]["9"]["success_rate"] == 1.0
    assert summary["per_layout_style"]["7_10"]["success_rate"] == 0.0
    assert summary["per_primary_object_category"]["mug"]["episodes"] == 2
    assert summary["per_task_style"]["task=CoffeeSetupMug|style_id=10"]["successes"] == 0
