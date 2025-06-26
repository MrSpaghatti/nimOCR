import tigr
import times # getTime is available through times.getTime() or system.getTime()
import strformat
import system # For system.getTime() as per tutorial context, though times.getTime() also works

# Import other modules from the project
import preprocessing # For FeaturePoint and later, actual preprocessing calls
import inference   # For runInference

const
  WindowWidth = 800
  WindowHeight = 600

type
  Point* = object
    x*: float
    y*: float
    timestamp*: int64 # Milliseconds since epoch
    pressure*: float # Normalized pressure (0.0 to 1.0, placeholder for now)

  Stroke* = seq[Point]
  DigitalInk* = seq[Stroke]

# Helper proc to convert Stroke to seq[FeaturePoint]
# This is a simplified version for the boilerplate.
# In a full app, this would likely live in preprocessing.nim and do more.
proc toFeatures(stroke: Stroke): seq[preprocessing.FeaturePoint] =
  # For now, just convert Point to FeaturePoint without full normalization/resampling
  # The tutorial's Part 5 ui.nim calls `normalizeStroke` first, then converts.
  # We'll follow that structure conceptually.

  # 1. Normalize (using the stub from preprocessing.nim)
  # The targetSize 20.0 is arbitrary for now, taken from tutorial's example.
  # The actual MNIST model expects 28x28, so this part will need refinement.
  let normalizedStrokePoints = preprocessing.normalizeStroke(stroke, 20.0)

  # 2. Convert Point to FeaturePoint
  for p in normalizedStrokePoints:
    result.add(preprocessing.FeaturePoint(x: p.x, y: p.y))

  # If preprocessing.normalizeStroke is just a stub returning original,
  # this will effectively just convert types.
  # If normalizedStrokePoints is empty, result will be empty.

proc runApplication*() =
  var screen = window(WindowWidth, WindowHeight, "Nim Handwriting Recognition", 0)

  # Application state
  var handwriting = newSeq[Stroke]()
  var currentStroke = newSeq[Point]()
  var isDrawing = false
  var recognizedText = "Draw a digit (0-9)" # Text to display

  # Load a font for displaying text
  let font = tfont() # Default TIGR font

  while screen.closed() == 0:
    # 1. Handle Input and Update State
    let mouseX = screen.mouseX().float
    let mouseY = screen.mouseY().float

    if screen.mouseDown(TIGR_MOUSE_LEFT):
      if not isDrawing:
        # Mouse button was just pressed: start a new stroke
        isDrawing = true
        currentStroke = newSeq[Point]() # Clear any previous data
        # Use system.getTime().toUnixMillis() as per tutorial context
        let p = Point(x: mouseX, y: mouseY, timestamp: system.getTime().toUnixMillis(), pressure: 1.0)
        currentStroke.add(p)
      else:
        # Mouse is being dragged: add a point to the current stroke
        let p = Point(x: mouseX, y: mouseY, timestamp: system.getTime().toUnixMillis(), pressure: 1.0)
        currentStroke.add(p)
    else:
      if isDrawing:
        # Mouse button was just released: finalize the stroke
        isDrawing = false
        if currentStroke.len > 1:
          handwriting.add(currentStroke)

          # TRIGGER THE RECOGNITION PIPELINE (Conceptual based on Tutorial Part 5)
          # Convert currentStroke to features
          let features = toFeatures(currentStroke)

          if features.len > 0:
            # The model path is relative to where the executable is run.
            # If running from src/: ../models/mnist-12.onnx
            # If executable is in project root: models/mnist-12.onnx
            # The README suggests running from src/, so ../models/ is appropriate.
            let digit = inference.runInference("../models/mnist-12.onnx", features)
            if digit != -1:
              recognizedText = &"Recognized: {digit}"
            else:
              recognizedText = "Could not recognize."
          else:
            recognizedText = "Not enough data for recognition."

        currentStroke = newSeq[Point]() # Reset for the next stroke

    # 2. Render the screen
    screen.clear(RGB(20, 20, 30)) # Dark blue-gray

    # Draw all completed strokes in gray
    for stroke in handwriting:
      if stroke.len >= 2:
        for i in 0 ..< stroke.len - 1:
          screen.line(stroke[i].x.int, stroke[i].y.int, stroke[i+1].x.int, stroke[i+1].y.int, RGB(100, 100, 100))

    # Draw the current, active stroke in white
    if currentStroke.len >= 2:
      for i in 0 ..< currentStroke.len - 1:
        screen.line(currentStroke[i].x.int, currentStroke[i].y.int, currentStroke[i+1].x.int, currentStroke[i+1].y.int, RGB(255, 255, 255))

    # Display the recognized text
    screen.print(font, 10, 10, RGB(255, 255, 0), recognizedText)

    # Display instructions
    screen.print(font, 10, WindowHeight - 30, RGB(150, 150, 150), "Draw a digit (0-9). Release mouse to recognize.")
    screen.print(font, 10, WindowHeight - 15, RGB(150, 150, 150), "Press any key to clear.")


    # Add a clear screen function
    if screen.key() != 0: # Check if any key was pressed
      handwriting = newSeq[Stroke]()
      currentStroke = newSeq[Point]() # Also clear current stroke if any
      isDrawing = false # Ensure drawing state is reset
      recognizedText = "Draw a digit (0-9)" # Reset text

    screen.update()
