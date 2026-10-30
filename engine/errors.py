"""Engine error types. The engine raises only these."""


class InputError(Exception):
    """The caller supplied an invalid or unsupported input."""


class StageError(Exception):
    """A pipeline stage failed."""

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason

    def __reduce__(self):
        return (type(self), (self.stage, self.reason))
