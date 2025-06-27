import tigr
import times # getTime is available through times.getTime() or system.getTime()
import strformat
import system # For system.getTime() as per tutorial context, though times.getTime() also works

# Import other modules from the project
import types       # For Point, Stroke, DigitalInk
import preprocessing # For FeaturePoint and later, actual preprocessing calls
import inference   # For runInference

const
  WindowWidth = 800
  WindowHeight = 600

# Helper procedure to process a stroke for TrOCR inference.
proc processStrokeForTrOCR*(stroke: types.Stroke, targetDim: int, lineThickness: int = 3): seq[float32] =
  # 1. Rasterize stroke to a 3-channel RGB uint8 image (0-255).
  #    normalizeStroke is called within rasterizeStrokeToRGB.
  let rgbImageData = preprocessing.rasterizeStrokeToRGB(stroke, targetDim, lineThickness)

  # 2. Convert to float32 tensor, rescale [0,1], normalize [-1,1], and set CHW format.
  result = preprocessing.prepareImageTensor(rgbImageData, targetDim, targetDim, 3)

proc runApplication*() =
  var screen = window(WindowWidth, WindowHeight, "Nim Handwriting Recognition", 0)

  # Application state
  var handwriting = newSeq[types.Stroke]()
  var currentStroke = newSeq[types.Point]()
  var isDrawing = false
  var recognizedText = "Draw some text..." # Updated default text
  let font = tfont()

  # Load TrOCR tokenizer data and decoder configuration
  # Define paths relative to the src directory where main.nim is expected to be run
  let modelsBaseDir = "../models/trocr/"
  let vocabPath = modelsBaseDir & "vocab.json"
  let genConfigPath = modelsBaseDir & "generation_config.json"
  let mainConfigPath = modelsBaseDir & "config.json" # Main config might have some fallbacks

  let (idToTokenMap, decoderConfig) = inference.loadTrOCTokenizerData(
      vocabPath, genConfigPath, mainConfigPath
  )

  if idToTokenMap.len == 0:
    echo "FATAL: Failed to load TrOCR tokenizer data. Check paths and file integrity."
    recognizedText = "Error: Tokenizer failed to load. Check console."
    # Allow window to run to show error, but inference won't work.

  let encoderPath = modelsBaseDir & "encoder_model.onnx"
  let decoderPath = modelsBaseDir & "decoder_model.onnx" # Or decoder_model_merged.onnx

  while screen.closed() == 0:
    let mouseX = screen.mouseX().float
    let mouseY = screen.mouseY().float

    if screen.mouseDown(TIGR_MOUSE_LEFT):
      if not isDrawing:
        isDrawing = true
        currentStroke = newSeq[types.Point]()
        currentStroke.add(types.Point(x: mouseX, y: mouseY, timestamp: system.getTime().toUnixMillis(), pressure: 1.0))
      else:
        currentStroke.add(types.Point(x: mouseX, y: mouseY, timestamp: system.getTime().toUnixMillis(), pressure: 1.0))
    else:
      if isDrawing:
        isDrawing = false
        if currentStroke.len > 1:
          handwriting.add(currentStroke)

          if idToTokenMap.len == 0: # Check if tokenizer loaded
             recognizedText = "Tokenizer error. Cannot recognize."
          else:
            const trocrGridSize = 384
            const lineThickness = 3 # Adjust as needed
            echo "Processing stroke for TrOCR..."
            let features = processStrokeForTrOCR(currentStroke, trocrGridSize, lineThickness)

            if features.len == trocrGridSize * trocrGridSize * 3:
              echo "Running TrOCR inference..."
              recognizedText = inference.runTrOCRInference(
                  encoderPath, decoderPath, features, idToTokenMap, decoderConfig
              )
              if recognizedText.startsWith("Error:"):
                echo recognizedText // Log error to console
              else:
                echo &"Recognized: {recognizedText}"
            else:
              recognizedText = "Preprocessing error."
              echo &"Error: Feature length mismatch. Expected {trocrGridSize*trocrGridSize*3}, got {features.len}"

        currentStroke = newSeq[types.Point]()

    screen.clear(RGB(20, 20, 30))

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
    screen.print(font, 10, WindowHeight - 30, RGB(150, 150, 150), "Draw text. Release mouse to recognize.")
    screen.print(font, 10, WindowHeight - 15, RGB(150, 150, 150), "Press any key to clear.")


    # Add a clear screen function
    if screen.key() != 0: # Check if any key was pressed
      handwriting = newSeq[types.Stroke]()
      currentStroke = newSeq[types.Point]() # Also clear current stroke if any
      isDrawing = false # Ensure drawing state is reset
      recognizedText = "Draw some text..." # Reset text

    screen.update()
