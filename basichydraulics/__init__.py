__version__ = "0.1.1"

from .channel import (
    Channel,
    Flat,
    TriangularChannel,
    RectangularChannel,
    TrapezoidalChannel,
    SemiCircularChannel,
    IrregularChannel,
    plot_channel,
)

from .structure import (
    Structure,
    SmallOpening,
    LargeOpening,
    plot_structure,
    g,
)

__all__ = [
    "Channel",
    "Flat",
    "TriangularChannel",
    "RectangularChannel",
    "TrapezoidalChannel",
    "SemiCircularChannel",
    "IrregularChannel",
    "plot_channel",
    "Structure",
    "SmallOpening",
    "LargeOpening",
    "plot_structure",
    "g",
]
