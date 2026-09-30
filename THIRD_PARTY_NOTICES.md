# Third-party notices

Original code, configurations and documentation contributed by the WorldToken
authors are licensed under the [MIT License](LICENSE), except where otherwise
noted. Third-party material retains its original copyright and license. The
WorldToken MIT license does not replace the licenses described below.

## Qwen2-VL rotary embeddings (Apache-2.0)

`worldtoken/encoder/rope2d.py` includes the `rotate_half`,
`apply_rotary_pos_emb_vision` and `VisionRotaryEmbedding` definitions from
Hugging Face Transformers' Qwen2-VL implementation. The 2D rotary-position
construction and vision-attention adaptation in that file also follow Qwen2-VL.
These upstream portions and their adaptations remain under Apache-2.0;
WorldToken's original additions are under MIT.

Reference source containing the copied definitions:
[Transformers v4.46.3, modeling_qwen2_vl.py](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py).

Upstream copyright:

> Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

The upstream module also credits EleutherAI's GPT-NeoX library and the GPT-NeoX
and OPT implementations in Transformers. WorldToken uses the rotary primitives
inside global attention across cameras, language and proprioception, instead
of Qwen2-VL's per-image attention, and adds its own query/key normalization and
temperature controls. See the source comments for the copied block and changes.

The full license is provided in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
Retain these notices and that license when redistributing the covered material.

## robomimic baseline patch (MIT)

`experiments/04_robocasa_scaling/baseline/robomimic.patch` contains excerpts from
and changes to the RoboCasa branch of robomimic at commit
`271a76c2d55c8b0f94d3d589f26fcae0d47f64a1`.

[Upstream source and license](https://github.com/ARISE-Initiative/robomimic/tree/271a76c2d55c8b0f94d3d589f26fcae0d47f64a1).

> Copyright (c) 2021 Stanford Vision and Learning Lab

The full upstream MIT license is provided in
[licenses/MIT-robomimic.txt](licenses/MIT-robomimic.txt). WorldToken's changes
provide library compatibility, shared language encoding and predictable output
directories. WorldToken-authored patch additions are covered by the root MIT
license; the original excerpts retain the upstream copyright above.

## Separately supplied dependencies and research assets

External libraries and simulators are installed separately and retain their own
licenses. The code license does not grant rights to separately distributed
model weights, expert demonstrations, experiment records, videos or simulator
assets; consult the licenses accompanying those releases.
