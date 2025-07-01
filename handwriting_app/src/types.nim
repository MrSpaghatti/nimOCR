# This module contains common type definitions used across the
# handwriting recognition application.

type
  Point* = object
    x*: float
    y*: float
    timestamp*: int64 # Milliseconds since epoch
    pressure*: float # Normalized pressure (0.0 to 1.0, placeholder for now)

  Stroke* = seq[Point]

  DigitalInk* = seq[Stroke]
