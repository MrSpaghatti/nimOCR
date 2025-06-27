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
  var recognizedText = "Loading models & tokenizer..." # Initial text
  let font = tfont()
  screen.print(font, 10, 10, RGB(255, 255, 0), recognizedText)
  screen.update() # Show loading message immediately


  # Load TrOCR tokenizer data and decoder configuration
  # Define paths relative to the src directory where main.nim is expected to be run
  let modelsBaseDir = "../models/trocr/"
  let vocabPath = modelsBaseDir & "vocab.json"
  let genConfigPath = modelsBaseDir & "generation_config.json"
  let mainConfigPath = modelsBaseDir & "config.json"
  let tokenizerConfigPath = modelsBaseDir & "tokenizer_config.json" # Added for max_length

  echo &"Attempting to load tokenizer files from base directory: {modelsBaseDir}"
  echo &"  Vocab path: {vocabPath}"
  echo &"  Gen Config path: {genConfigPath}"
  echo &"  Main Config path: {mainConfigPath}"
  echo &"  Tokenizer Config path: {tokenizerConfigPath}"

  let (idToTokenMap, decoderConfig, tokenizerSuccess) = inference.loadTrOCTokenizerData(
      vocabPath, genConfigPath, mainConfigPath, tokenizerConfigPath
  )

  if not tokenizerSuccess:
    recognizedText = "FATAL: Tokenizer/Config load error. Check console."
    # Update screen and then proceed to loop to keep window responsive
    screen.clear(RGB(20,20,30))
    screen.print(font, 10, 10, RGB(255,0,0), recognizedText) // Error in red
    screen.update()
  else:
    recognizedText = "Draw some text..." # Ready message

  let encoderPath = modelsBaseDir & "encoder_model.onnx"
  let decoderPath = modelsBaseDir & "decoder_model.onnx" # Or decoder_model_merged.onnx
  echo &"Encoder model path: {encoderPath}"
  echo &"Decoder model path: {decoderPath}"


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

          if not tokenizerSuccess: # Check if tokenizer loaded successfully
             recognizedText = "Tokenizer error. Cannot recognize."
             # No screen.update() here, error is already on screen from startup
          else:
            recognizedText = "Processing..."
            # Force redraw to show "Processing..."
            # This requires clearing, drawing strokes, then printing new text and updating
            screen.clear(RGB(20,20,30))
            for strokeDraw in handwriting: # Draw existing strokes
              if strokeDraw.len >= 2:
                for i in 0 ..< strokeDraw.len - 1:
                  screen.line(strokeDraw[i].x.int, strokeDraw[i].y.int, strokeDraw[i+1].x.int, strokeDraw[i+1].y.int, RGB(100,100,100))
            # Don't draw currentStroke as it's now part of handwriting / being processed
            screen.print(font, 10, 10, RGB(255, 255, 0), recognizedText) // Show "Processing..."
            screen.print(font, 10, WindowHeight - 30, RGB(150,150,150), "Draw text. Release mouse to recognize.")
            screen.print(font, 10, WindowHeight - 15, RGB(150,150,150), "Press any key to clear.")
            screen.update()


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
    var displayText = recognizedText
    const maxDisplayChars = (WindowWidth - 20) div 8 // Approx chars that fit, 8px/char, 10px margin each side
    if displayText.len > maxDisplayChars:
      displayText = displayText[0 ..< maxDisplayChars - 3] & "..."
    screen.print(font, 10, 10, RGB(255, 255, 0), displayText)

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
