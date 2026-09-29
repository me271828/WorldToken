"""Configuration compatibility for the paper's original four-token run."""

from worldtoken.transformer.frame_major import FrameMajorContinuousTokenTransformer


class MultiTokenContinuousTokenTransformer(FrameMajorContinuousTokenTransformer):
    """Accept the original parameter name without changing state-dict keys."""

    def __init__(self, *, retained_tokens_per_frame: int, **kwargs) -> None:
        super().__init__(tokens_per_frame=retained_tokens_per_frame, **kwargs)
        self.retained_tokens_per_frame = int(retained_tokens_per_frame)
