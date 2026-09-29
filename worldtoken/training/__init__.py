"""Training-side helpers extracted from the BC entrypoint.

These are the self-contained pieces of ``train_bc`` (action-normalizer fitting,
holdout metric aggregation, eval-batch materialization) that carry no dependency
on the CLI ``argparse`` namespace or the training loop, so they can be reused and
unit-tested in isolation. The thin orchestration stays in ``train_bc.py``.
"""
