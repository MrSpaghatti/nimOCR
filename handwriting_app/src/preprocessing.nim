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
    result = newSeq[types.Point](1) # Corrected from ui.Point
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

  result = newSeq[types.Point](stroke.len) # Corrected from ui.Point
  for i, p in stroke:
    result[i] = p # Copy timestamp and pressure
    # Apply translation to move the stroke's bounding box origin (minX, minY) to (0,0)
    # and then apply scaling.
    # Max coordinate will be `maxDim * scaleFactor = maxDim * (scaleTo / maxDim) = scaleTo`.
    var newX = (p.x - minX) * scaleFactor
    var newY = (p.y - minY) * scaleFactor

    result[i].x = newX
    result[i].y = newY

proc rasterizeStrokeToBinaryGrid*(stroke: types.Stroke, gridSize: int, lineThickness: int = 3): seq[bool] =
  ## Renders a normalized stroke onto a square binary grid.
  ## Input stroke is assumed to be normalized, with coordinates in [0, gridSize-1].
  ## Output is a flattened sequence of booleans (true for foreground/stroke).

  result = newSeq[bool](gridSize * gridSize) # Initialize grid to all false (background)

  if stroke.len < 1:
    return result

  # Helper to mark a pixel and its neighbors for thickness
  proc markPixelWithThickness(x, y: int) =
    for offsetY in -lineThickness div 2 .. lineThickness div 2:
      for offsetX in -lineThickness div 2 .. lineThickness div 2:
        let finalX = x + offsetX
        let finalY = y + offsetY
        if finalX >= 0 and finalX < gridSize and finalY >= 0 and finalY < gridSize:
          result[finalY * gridSize + finalX] = true

  if stroke.len == 1:
    markPixelWithThickness(stroke[0].x.int, stroke[0].y.int)
    return result

  for i in 0 ..< stroke.len - 1:
    let p1 = stroke[i]
    let p2 = stroke[i+1]

    let x1f = p1.x
    let y1f = p1.y
    let x2f = p2.x
    let y2f = p2.y

    let dx = x2f - x1f
    let dy = y2f - y1f

    # Determine number of steps for line interpolation
    # Ensure enough steps to cover the distance, considering thickness.
    let len = sqrt(dx*dx + dy*dy)
    let steps = if len > 0: int(len * 1.5) else: 1 # Factor 1.5 for density, at least 1 step.
                                                 # Adjust factor as needed.

    for step in 0 .. steps:
      let t = if steps == 0: 0.0 else: float(step) / float(steps)
      let currentX = x1f + t * dx
      let currentY = y1f + t * dy
      markPixelWithThickness(currentX.round.int, currentY.round.int)
  return result

proc rasterizeStrokeToRGB*(stroke: types.Stroke, targetDim: int, lineThickness: int = 3): seq[uint8] =
  ## Normalizes a stroke and rasterizes it to a 3-channel RGB image (seq[uint8]).
  ## Stroke is drawn as black (0,0,0) on a white (255,255,255) background.

  let normalizedStroke = normalizeStroke(stroke, targetDim.float)
  let binaryGrid = rasterizeStrokeToBinaryGrid(normalizedStroke, targetDim, lineThickness)

  result = newSeq[uint8](targetDim * targetDim * 3)
  for i in 0 ..< binaryGrid.len:
    let pixelVal = if binaryGrid[i]: 0'u8 else: 255'u8 // Black stroke on white BG
    result[i * 3 + 0] = pixelVal # R
    result[i * 3 + 1] = pixelVal # G
    result[i * 3 + 2] = pixelVal # B
  return result

proc prepareImageTensor*(imageData: seq[uint8], width: int, height: int, numChannels: int = 3): seq[float32] =
  ## Converts uint8 RGB image data to a float32 CHW tensor,
  ## rescaling to [0,1] and normalizing to [-1,1].
  assert imageData.len == width * height * numChannels, "Image data size mismatch"

  result = newSeq[float32](width * height * numChannels)
  let rescaleFactor = 1.0f / 255.0f
  let imageMean = [0.5f, 0.5f, 0.5f] # Per-channel mean
  let imageStd = [0.5f, 0.5f, 0.5f]  # Per-channel std

  for c in 0 ..< numChannels:
    for y in 0 ..< height:
      for x in 0 ..< width:
        let srcIdx = (y * width + x) * numChannels + c
        let pixelUint8 = imageData[srcIdx]

        let rescaledPixel = pixelUint8.float32 * rescaleFactor
        let normalizedPixel = (rescaledPixel - imageMean[c]) / imageStd[c]

        # CHW format: result[channel * H * W + row * W + col]
        result[c * height * width + y * width + x] = normalizedPixel
  return result

# Delete the old MNIST-specific rasterizeStroke
# proc rasterizeStroke*(stroke: types.Stroke, gridSize: int): seq[float] = ... (old implementation)
