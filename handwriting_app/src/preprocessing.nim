import math
# We need to import ui to access Point and Stroke types.
# This creates a potential circular dependency if ui also imports preprocessing.
# For a boilerplate, we'll proceed, but this might need refactoring in a full implementation
# e.g. by moving Point/Stroke to a separate types module.
import types # Changed from 'ui'

# FeaturePoint type is no longer needed as rasterizeStroke produces seq[float]
# and inference.nim now accepts seq[float].

proc normalizeStroke*(stroke: types.Stroke, targetSize: float): types.Stroke =
  ## Normalizes stroke coordinates to a target bounding box (e.g., 28x28).
  ## The output points will be within the range [0, targetSize-1] or similar,
  ## scaled to fit, with the top-left of the bounding box at (0,0).
  if stroke.len == 0: return @[]

  # Find the current bounding box of the stroke
  var minX = high(float)
  var minY = high(float)
  var maxX = low(float)
  var maxY = low(float)

  for p in stroke:
    minX = min(minX, p.x)
    minY = min(minY, p.y)
    maxX = max(maxX, p.x)
    maxY = max(maxY, p.y)

  let width = maxX - minX
  let height = maxY - minY

  # Handle single-point strokes or strokes on a single line.
  # If width or height is 0, it means all points are collinear.
  # We want to avoid division by zero in `scale`.
  # A single point (width=0, height=0) should be placed, e.g., in the middle of the target.
  # A line (width=0 or height=0) should be scaled along its non-zero dimension.

  let N = targetSize
  # scaleTo is the target maximum coordinate value, e.g., 27.0 for targetSize 28 (N=28).
  # If targetSize (N) is 1.0, scaleTo is 0.0.
  # If targetSize (N) is < 1.0 (e.g. 0.5), scaleTo is 0.0 (points map to origin).
  let scaleTo = if N >= 1.0: N - 1.0 else: 0.0

  if width == 0 and height == 0: # Single point
    result = newSeq[ui.Point](1)
    result[0] = stroke[0] # Copy timestamp and pressure
    # Place single point in the center of the [0, scaleTo] range.
    # If scaleTo is 0.0 (targetSize=1), point becomes (0.0, 0.0).
    # If scaleTo is 27.0 (targetSize=28), point becomes (13.5, 13.5).
    result[0].x = scaleTo / 2.0
    result[0].y = scaleTo / 2.0
    return result

  var scaleFactor: float
  # maxDim is guaranteed to be > 0 here because the single point case (width=0, height=0) was handled above.
  let maxDim = max(width, height)

  if scaleTo <= 0: # Handles targetSize <= 1. All points map to (0,0).
                  # scaleTo being 0 means N=1. maxDim > 0. scaleFactor becomes 0.
    scaleFactor = 0.0
  else:
    # maxDim is > 0 here.
    scaleFactor = scaleTo / maxDim

  result = newSeq[ui.Point](stroke.len)
  for i, p in stroke:
    result[i] = p # Copy timestamp and pressure
    # Apply translation to move the stroke's bounding box origin (minX, minY) to (0,0)
    # and then apply scaling.
    # Max coordinate will be `maxDim * scaleFactor = maxDim * (scaleTo / maxDim) = scaleTo`.
    var newX = (p.x - minX) * scaleFactor
    var newY = (p.y - minY) * scaleFactor

    result[i].x = newX
    result[i].y = newY

proc rasterizeStroke*(stroke: ui.Stroke, gridSize: int): seq[float] =
  ## Renders a normalized stroke onto a square grid of gridSize x gridSize.
  ## The input stroke is assumed to be normalized, with coordinates in [0, gridSize-1].
  ## Output is a flattened sequence of floats (pixel intensities, 0.0 or 1.0).

  result = newSeq[float](gridSize * gridSize) # Initialize grid to all 0.0

  if stroke.len < 1:
    return result # Return empty grid if no points

  # For a single point stroke, mark the corresponding cell.
  if stroke.len == 1:
    let p = stroke[0]
    let x = p.x.int
    let y = p.y.int
    if x >= 0 and x < gridSize and y >= 0 and y < gridSize:
      result[y * gridSize + x] = 1.0
    return result

  # "Draw" lines between consecutive points.
  # Using a simple line drawing by oversampling. Not as precise as Bresenham's.
  for i in 0 ..< stroke.len - 1:
    let p1 = stroke[i]
    let p2 = stroke[i+1]

    let x1 = p1.x
    let y1 = p1.y
    let x2 = p2.x
    let y2 = p2.y

    let dx = x2 - x1
    let dy = y2 - y1
    let steps = max(abs(dx), abs(dy)) * 2.0 # Oversample for denser line; factor of 2 is arbitrary
                                          # Or use a fixed number of steps, e.g., 100 per segment.
                                          # Let's make it proportional to length for now.

    if steps == 0: # p1 and p2 are the same point
      let x = x1.int
      let y = y1.int
      if x >= 0 and x < gridSize and y >= 0 and y < gridSize:
        result[y * gridSize + x] = 1.0
      continue

    for step in 0 .. int(steps):
      let t = float(step) / float(steps)
      let currentX = x1 + t * dx
      let currentY = y1 + t * dy

      let cellX = currentX.int
      let cellY = currentY.int

      if cellX >= 0 and cellX < gridSize and cellY >= 0 and cellY < gridSize:
        result[cellY * gridSize + cellX] = 1.0 # Mark cell as 1.0 (white)

  # Simple thickness: also mark neighboring pixels for each marked pixel
  # This is a very crude way to add thickness. A better way would be to draw thicker lines.
  # For now, let's skip this to keep it simple. If lines are too thin, this can be added.
  # Example of simple 3x3 dilation (would need a copy of result):
  # var tempResult = result # copy
  # for r in 1 ..< gridSize - 1:
  #   for c in 1 ..< gridSize - 1:
  #     if result[r * gridSize + c] == 1.0:
  #       for dr in -1..1:
  #         for dc in -1..1:
  #           tempResult[(r+dr) * gridSize + (c+dc)] = 1.0
  # result = tempResult

  return result

# The old resampleStroke stub is no longer needed.
# rasterizeStroke provides the functionality of converting a stroke to a feature vector (bitmap).
