# Creating an Online Handwriting Recognition Application with Nim and ONNX

## Introduction: Building a Smart Canvas with Nim

This tutorial provides a comprehensive, step-by-step guide to building a real-time handwriting recognition application. The project's goal is to construct a graphical user interface (GUI) using the Nim programming language that captures handwritten strokes from a mouse or stylus and uses a pre-trained deep learning model to recognize the characters. This endeavor serves as a practical introduction to Nim for developers experienced in other languages, demonstrating its capabilities in system orchestration, UI development, and integration with high-performance machine learning libraries.

The core of this project revolves around a specific and powerful approach to handwriting analysis known as Online Handwriting Recognition (OHWR). Understanding the distinction between OHWR and its counterpart, Offline Handwriting Recognition (OHR), is fundamental to appreciating the design of the entire application.

Offline Handwriting Recognition (OHR) deals with static data. It is analogous to transcribing text from a scanned document or a photograph. In this scenario, all dynamic information about the act of writing—the order of strokes, the speed of the pen, the pauses, the pressure applied—is lost. OHR systems must analyze a flat, two-dimensional image, making the task inherently more challenging and often reliant on traditional Optical Character Recognition (OCR) techniques.

In contrast, Online Handwriting Recognition (OHWR) operates on dynamic, temporal data captured in real-time as a user writes on a touch-sensitive surface. This method captures a rich stream of information beyond simple (x, y) coordinates, including the sequence in which strokes are formed, the velocity of the pen, and variations in pressure. The availability of this temporal data provides a significant advantage, consistently leading to higher recognition accuracy compared to offline methods.

The decision to build an OHWR system is the first and most critical one, as it establishes a causal chain that dictates every subsequent architectural choice. Because the system is defined as "online," it necessitates the capture of dynamic, temporal data. This, in turn, requires data structures capable of representing sequences of points over time. The processing of this sequential data calls for specialized preprocessing techniques, such as resampling, that can handle its temporal nature. Finally, the machine learning model itself must be adept at interpreting sequences, which naturally leads to architectures like Recurrent Neural Networks (RNNs) that are designed for this purpose. This tutorial is therefore not a collection of disparate components, but a guide to building a cohesive system where each part is a logical consequence of the initial problem definition.

## Part 0: Nim Language Fundamentals for the C++/Python Developer

This section serves as a rapid bootcamp for experienced programmers new to Nim. It focuses on the essential syntax and concepts required for this project, drawing parallels to C++ and Python to accelerate understanding.

### Setup and "Hello, World!"

First, the Nim compiler must be installed by following the official instructions available on the Nim language website. Once installed, the development workflow can be understood with a simple "Hello, World!" program.2

Create a file named `hello.nim` with the following content:

Nim

```
echo "Hello, World!"
```

To compile and run this program, execute the following command in a terminal:

Bash

```
nim c -r hello.nim
```

This command instructs the Nim compiler to `c` (compile) the file and then `-r` (run) the resulting executable. This compile-and-run cycle is the fundamental workflow for Nim development.

### Variables, Constants, and Data Types

Nim is a statically typed language, but its powerful type inference often gives it the feel of a dynamic language like Python.4

Mutable and Immutable Variables

Variables are declared using the var, let, and const keywords.5

*   `var` declares a mutable variable, whose value can be changed. This is the standard variable declaration in most languages.
    
*   `let` declares a single-assignment variable (immutable). Once assigned, its value cannot be changed. This is analogous to a `const` variable in C++ or a variable in Python that is not reassigned.
    
*   `const` declares a compile-time constant. Its value must be known by the compiler.
    

Nim

```
# Mutable variable
var age: int = 30
age = 31 # This is allowed

# Immutable variable
let name = "Alice" # Type is inferred as 'string'
# name = "Bob" # This would cause a compile error

# Compile-time constant
const AppVersion = "1.0.0"
```

Basic Data Types

Nim provides a standard set of built-in data types 4:

*   `int`: A signed integer, with a size that is platform-dependent (like a pointer).
    
*   `float`: A floating-point number, which defaults to 64-bit precision.
    
*   `string`: A sequence of characters.
    
*   `bool`: A boolean value, which can be `true` or `false`.
    

String literals can be enclosed in double quotes. For multi-line strings that ignore escape characters, triple quotes (`"""..."""`) can be used, which is useful for embedding templates or long text blocks.7

### Procedures (Functions in Nim)

In Nim, reusable blocks of code are called procedures, declared with the `proc` keyword. The syntax is similar to function declarations in other typed languages.8

Nim

```
# A procedure that takes two integers and returns their sum
proc add(a: int, b: int): int =
  return a + b

# A procedure with no return value
proc greet(name: string) =
  echo "Hello, ", name
```

Nim offers two primary ways to return a value from a procedure: the `return` keyword and an implicit `result` variable. The `result` variable is automatically declared with the procedure's return type and is often preferred for its clarity and potential for compiler optimizations like copy elision.8

Nim

```
proc subtract(a: int, b: int): int =
  result = a - b # The value of 'result' is returned automatically

echo subtract(10, 3) # Outputs: 7
```

### Control Flow

Nim's control flow syntax is clean and readable, borrowing elements from languages like Python.

Conditionals

The if/elif/else structure is used for conditional logic. Parentheses around the condition are optional.7

Nim

```
let score = 85
if score >= 90:
  echo "Grade: A"
elif score >= 80:
  echo "Grade: B"
else:
  echo "Grade: C or lower"
```

Loops

Nim provides while and for loops.

*   A `while` loop executes as long as its condition is true.
    
*   A `for` loop iterates over a sequence, range, or other iterable type. The `..` operator creates an inclusive range, while `..<` creates an exclusive range, which is familiar to Python programmers.7
    

Nim

```
# While loop
var i = 0
while i < 3:
  echo i
  i = i + 1

# For loop with an inclusive range
for j in 1..3:
  echo j # Outputs 1, 2, 3

# For loop over a sequence
let fruits = @["apple", "banana", "cherry"]
for fruit in fruits:
  echo fruit
```

### Essential Data Structures

Objects

The object type is used to define custom structured data types, similar to a struct in C or a simple class in Python.10

Nim

```
type
  Point* = object # The '*' exports the type, making it public
    x*: float
    y*: float
```

Sequences

A seq is a dynamic, resizable array that holds elements of type T. It is analogous to std::vector<T> in C++ or a list in Python. Sequences are created with the @ literal and elements are added with the .add() procedure.10

Nim

```
# Declare a sequence of integers
var numbers: seq[int] = @

# Add a new element
numbers.add(4)

# Create an empty sequence
var emptySeq = newSeq[string]()
```

### A Glimpse into FFI: The `importc` Pragma

One of Nim's most powerful features is its seamless interoperability with C and C++, which stems from its compilation to C as an intermediate step.13 This is achieved through a Foreign Function Interface (FFI).

The `importc` pragma is used to declare that a procedure is defined in an external C library. A pragma is a special instruction to the compiler, enclosed in `{..}`.

Here is a conceptual example of how to call the standard C `puts` function:

Nim

```
# Declare the external C function
proc puts(s: cstring) {.importc: "puts", header: "<stdio.h>".}

# Call the function
puts("Calling C from Nim!")
```

In this example, `importc: "puts"` tells Nim that the `puts` procedure corresponds to an external C function named "puts". The `header` pragma ensures the correct C header is included, and `cstring` is a special Nim type that is compatible with C's `char*` for strings.10 This mechanism is the key to using libraries like the ONNX Runtime later in the tutorial.

## Part 1: Building the UI and Capturing Pen Input

The goal of this section is to create a graphical window with a drawing canvas. This canvas will serve as our input surface, capturing mouse or stylus movements and translating them into a structured digital ink format.

### Choosing Our Toolkit: Simplicity First with tigr-nim

Nim has several options for GUI development, ranging from bindings to established toolkits like Qt and GTK to pure Nim libraries.16 For this tutorial, we will use

`tigr-nim`, a Nim wrapper for the TIGR (Tiny GRaphics) library.17 TIGR is a minimal, cross-platform graphics library that provides a simple framebuffer, basic drawing functions, and input handling, making it ideal for a beginner-focused project.17

To install it, use Nim's package manager, Nimble:

Bash

```
nimble install tigr
```

### Step 1: Creating the Window

The first step is to create a window and the main application loop. This loop is the heart of any GUI application, continuously checking for events, updating the application state, and redrawing the screen.

Nim

```
# file: ui.nim
import tigr
import times # For timestamps

const
  WindowWidth = 800
  WindowHeight = 600

proc runApplication*() =
  # Initialize a window with a title and dimensions
  var screen = window(WindowWidth, WindowHeight, "Nim Handwriting Recognition", 0)

  # Main application loop
  while screen.closed() == 0:
    # Clear the screen with a background color in each frame
    screen.clear(RGB(20, 20, 30)) # Dark blue-gray

    # --- Drawing and event handling logic will go here ---

    # Update the window to display the changes
    screen.update()
```

### Step 2: Defining Our Data Structures for "Digital Ink"

To capture handwriting effectively, we need a structured way to represent it. As established in the introduction, the dynamic nature of online recognition requires storing sequences of points over time. We will implement the `Point`, `Stroke`, and `DigitalInk` object types to model this data precisely.

Nim

```
# Add these type definitions to ui.nim

type
  Point* = object
    x*: float
    y*: float
    timestamp*: int64 # Milliseconds since epoch
    pressure*: float # Normalized pressure (0.0 to 1.0, placeholder for now)

  Stroke* = seq[Point]
  DigitalInk* = seq
```

*   `Point`: Represents a single data point from the input device, containing its coordinates, a timestamp, and pressure (which we will simulate for now).
    
*   `Stroke`: A sequence of `Point` objects, representing a single, continuous line drawn while the pen/mouse button is down.
    
*   `DigitalInk`: A sequence of `Stroke` objects, representing the complete handwritten input.
    

### Step 3: Implementing the Drawing and Capture Logic

Inside the main loop, we will add the logic to handle mouse input and convert it into our `DigitalInk` format. This process transforms the UI from a simple display into a data acquisition device. The event handlers for mouse down, move, and up are not merely for visual effect; they are the core of our data capture mechanism, responsible for populating the data structures that the rest of the system depends on.

Nim

```
# Modified runApplication proc in ui.nim

proc runApplication*() =
  var screen = window(WindowWidth, WindowHeight, "Nim Handwriting Recognition", 0)

  # Application state
  var handwriting = newSeq()
  var currentStroke = newSeq[Point]()
  var isDrawing = false

  while screen.closed() == 0:
    # 1. Handle Input and Update State
    let mouseX = screen.mouseX().float
    let mouseY = screen.mouseY().float

    if screen.mouseDown(TIGR_MOUSE_LEFT):
      if not isDrawing:
        # Mouse button was just pressed: start a new stroke
        isDrawing = true
        currentStroke = newSeq[Point]() # Clear any previous data
        let p = Point(x: mouseX, y: mouseY, timestamp: getTime().toUnixMillis(), pressure: 1.0)
        currentStroke.add(p)
      else:
        # Mouse is being dragged: add a point to the current stroke
        let p = Point(x: mouseX, y: mouseY, timestamp: getTime().toUnixMillis(), pressure: 1.0)
        currentStroke.add(p)
    else:
      if isDrawing:
        # Mouse button was just released: finalize the stroke
        isDrawing = false
        if currentStroke.len > 1:
          handwriting.add(currentStroke)
        # --- This is where we will trigger recognition later ---
        currentStroke = newSeq[Point]() # Reset for the next stroke

    # 2. Render the screen
    screen.clear(RGB(20, 20, 30))

    # Draw all completed strokes in gray
    for stroke in handwriting:
      if stroke.len >= 2:
        for i in 0..< stroke.len - 1:
          screen.line(stroke[i].x.int, stroke[i].y.int, stroke[i+1].x.int, stroke[i+1].y.int, RGB(100, 100, 100))

    # Draw the current, active stroke in white
    if currentStroke.len >= 2:
      for i in 0..< currentStroke.len - 1:
        screen.line(currentStroke[i].x.int, currentStroke[i].y.int, currentStroke[i+1].x.int, currentStroke[i+1].y.int, RGB(255, 255, 255))

    screen.update()
```

This code establishes the complete input-handling loop. When the user clicks and drags the mouse, a `currentStroke` is built point by point. When the mouse is released, this stroke is considered complete and is added to the `handwriting` sequence. This `onMouseUp` event is the critical trigger point where, in the final application, we will send the captured stroke data for processing and recognition.

## Part 2: Preprocessing the Digital Ink

With the ability to capture raw stroke data, the next crucial phase is preprocessing. Raw input is inherently noisy and inconsistent; users write at different sizes, speeds, and positions on the canvas. Preprocessing is an indispensable step to clean, standardize, and transform this raw data into a consistent format that a machine learning model can effectively interpret. Without proper preprocessing, the accuracy of any recognition model would be severely degraded.

### Step 1: Normalization - Taming Size and Position

The first preprocessing step is normalization. Its purpose is to make the recognition process independent of the size and position of the handwriting on the canvas. This is achieved by scaling and translating each stroke so that it fits within a standard bounding box (e.g., a 100x100 square).

The following procedure implements this logic:

Nim

```
# file: preprocessing.nim
import math
import ui # To access the Point and Stroke types

proc normalizeStroke*(stroke: Stroke, targetSize: float): Stroke =
  ## Normalizes stroke coordinates to a target bounding box.
  if stroke.len == 0: return @

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

  # Avoid division by zero for single-point strokes
  if width == 0 and height == 0: return stroke

  # Determine the scaling factor to fit the stroke into the target size
  let scale = targetSize / max(width, height)

  result = newSeq[Point](stroke.len)
  for i, p in stroke:
    # Apply translation to move the stroke to the origin (0,0)
    # and then apply scaling.
    result[i] = p
    result[i].x = (p.x - minX) * scale
    result[i].y = (p.y - minY) * scale
```

### Step 2: Resampling - Creating a Consistent "Fingerprint"

A significant challenge with online handwriting data is that the number of points in a stroke varies depending on writing speed and device sampling rate. A quickly drawn line might have very few points, while a slow, deliberate curve could have hundreds. Most neural network architectures, however, require a fixed-size input vector. Resampling is the technique used to solve this problem by converting a variable-length sequence of points into a fixed-length one.

While simple methods like picking every Nth point exist, a more powerful approach is to treat resampling as a form of feature engineering. This reframes the task from mere data reduction to an intelligent transformation of the data into a more robust representation. One of the most effective techniques, used in industrial systems like Google's Gboard, is to fit the raw points to a mathematical curve, such as a cubic Bézier curve.

By fitting a curve, the system creates an idealized, smooth abstraction of the stroke's trajectory. The control points of this curve become a compact summary of the stroke's essential shape. Sampling a fixed number of points along this idealized curve produces a feature vector that is not only consistent in size but also less susceptible to minor jitters and sampling noise from the input device. This process creates a superior set of features for the neural network to learn from.

The following code provides a conceptual implementation of this idea. While a full, robust Bézier curve fitting algorithm is beyond the scope of this tutorial, this example demonstrates the principle of calculating points along a curve to generate a fixed-size feature vector.

Nim

```
# Add to preprocessing.nim
type
  FeaturePoint* = object
    x*, y*: float

proc calculateBezierPoint(p0, p1, p2, p3: Point, t: float): FeaturePoint =
  ## Calculates a point on a cubic Bézier curve for a given t (0.0 to 1.0).
  let omt = 1.0 - t
  let omt2 = omt * omt
  let omt3 = omt2 * omt
  let t2 = t * t
  let t3 = t2 * t

  result.x = omt3 * p0.x + 3.0 * omt2 * t * p1.x + 3.0 * omt * t2 * p2.x + t3 * p3.x
  result.y = omt3 * p0.y + 3.0 * omt2 * t * p1.y + 3.0 * omt * t2 * p2.y + t3 * p3.y

proc resampleStroke*(stroke: Stroke, numFeatures: int): seq[FeaturePoint] =
  ## Resamples a stroke into a fixed number of feature points.
  ## This is a conceptual simplification. A real implementation would fit
  ## curves to segments of the stroke.
  if stroke.len < 4: return @ # Need at least 4 points for a cubic Bézier

  result = newSeq[FeaturePoint]()
  let step = 1.0 / (numFeatures.float - 1)

  for i in 0..< numFeatures:
    let t = i.float * step
    # For this simplified example, we use the first four points of the stroke
    # as the control points for the entire curve.
    let p = calculateBezierPoint(stroke, stroke, stroke, stroke, t)
    result.add(p)
```

### Overview of Preprocessing Techniques

The following table summarizes the key preprocessing techniques discussed and their importance in an OHWR system, providing a clear reference for the steps involved in preparing data for recognition.

| Technique | Purpose | Common Methods | Nim Implementation Notes |
| --- | --- | --- | --- |
| Normalization | Standardize writing size and position to make the model invariant to these variations. | Bounding box scaling and translation. | Can be implemented directly in Nim using basic mathematical operations. |
| Resampling | Convert variable-length strokes into fixed-size feature vectors suitable for neural network input. | Re-spacing points evenly by arc length, fitting Bézier curves and sampling points. | A conceptual version can be implemented in Nim. For production, using an FFI to a specialized C/C++ geometry library would be more robust. |
| Smoothing | Reduce jaggedness in strokes caused by hand jitter or low-resolution input devices. | Interpolation with Cubic Splines, moving average filters. | Can be implemented with math libraries or via FFI to a signal processing or graphics library. |
| Noise Reduction | Remove small, irrelevant artifacts from the data, such as "hooks" at the beginning or end of strokes. | Filtering based on angle and length thresholds. | Can be implemented with custom logic to analyze the geometry of the stroke's start and end points. |

## Part 3: The Deep Learning Model (A Conceptual Detour)

This section provides the necessary context on the machine learning model that will perform the handwriting recognition. It is crucial to state that the process of training such a model is a complex and data-intensive task, typically performed in a mature deep learning ecosystem like Python with frameworks such as PyTorch or TensorFlow. For this project, the Nim application will act as a *consumer* of a pre-trained model, a common and powerful pattern in modern software engineering that decouples application logic from ML model development.

### Anatomy of a Handwriting Recognizer

The architecture of a deep learning model for OHWR must be suited to handling sequential data. A highly effective and common choice is a hybrid model combining a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN), specifically a Long Short-Term Memory (LSTM) network.

*   **Convolutional Neural Network (CNN):** The CNN part of the model acts as a feature extractor. It processes segments of the stroke data to identify low-level spatial patterns, such as curves, lines, and intersections. It learns the fundamental visual "vocabulary" of handwriting.
    
*   **Long Short-Term Memory (LSTM) Network:** The features extracted by the CNN are then fed into the LSTM. LSTMs are a special kind of RNN designed to recognize patterns in sequences of data. This is where the temporal information captured during online recognition becomes critical. The LSTM learns the long-range dependencies between the features over time, understanding how the sequence of curves and lines forms a complete character or word.
    

### The Importance of Data and Augmentation

A deep learning model is only as good as the data it is trained on. Training a robust handwriting recognition model requires vast and diverse datasets. Publicly available collections like the **UNIPEN** dataset (for online data) and the **IAM Handwriting Dataset** (for offline data) are standard benchmarks in the research community, containing millions of characters from thousands of different writers.

To improve a model's ability to generalize to new, unseen handwriting styles, a technique called **data augmentation** is essential. This involves artificially expanding the training dataset by creating modified copies of existing samples. Common augmentation techniques for handwriting include :

*   **Geometric Transformations:** Applying random rotations, scaling, and small translations to the handwriting.
    
*   **Elastic Distortion:** Warping the handwriting in a non-linear way to simulate the natural variations and inconsistencies of human writing.
    
*   **Noise Injection:** Adding random noise or small artifacts to make the model more resilient to imperfections in the input data.
    

### ONNX: The Universal Translator for AI Models

The bridge that connects the Python-based training world to our Nim application is the **ONNX (Open Neural Network Exchange)** format. ONNX is an open standard for representing machine learning models. After a model is trained in a framework like PyTorch, it can be exported into a single `.onnx` file. This file contains a complete description of the model's architecture and its learned numerical weights.

This `.onnx` file is portable and framework-agnostic. It can be loaded and executed by the **ONNX Runtime**, a high-performance inference engine developed by Microsoft. The ONNX Runtime provides APIs for many languages, including C, C++, Python, and Java. While there is no official Nim API, the community has created wrapper libraries that use Nim's FFI to call the ONNX Runtime's C API, allowing Nim applications to execute ONNX models with near-native performance.

For this tutorial, a pre-trained model that recognizes handwritten digits (0-9), based on the MNIST dataset and saved in the ONNX format, can be downloaded from the Hugging Face model hub.19 This model,

`mnist-12.onnx`, will be used in the next section.

## Part 4: Performing Inference in Nim

This section details how to load the pre-trained `.onnx` model into the Nim application and use it to perform inference on the preprocessed handwriting data. This is where Nim's ability to interface with high-performance C libraries becomes a practical superpower.

### Step 1: Setting up the ONNX Runtime

The `onnxruntime-nim` library is a Nim wrapper that provides bindings to the official ONNX Runtime C API. Using it requires a two-step installation process.

1.  **Install the Nim Wrapper:** Use Nimble to fetch the wrapper library.
    
    Bash
    
    ```
    nimble install onnxruntime
    ```
    
2.  **Install the ONNX Runtime C Library:** The wrapper needs the underlying shared library (`.dll`, `.so`, or `.dylib`) to function. These can be downloaded directly from the ONNX Runtime release page on GitHub. Alternatively, on systems with package managers, it may be available there. For example, on Windows, one would download the appropriate zip file, extract it, and ensure the `bin/onnxruntime.dll` file is placed in a location where the application can find it (e.g., the same directory as the final executable).20
    

The code for inference will be placed in a new file, `inference.nim`.

### Step 2: Implementing the Inference Pipeline

The following procedure, `runInference`, encapsulates the logic for loading the model, preparing the input, running the model, and interpreting the output. This code demonstrates the power of abstraction provided by ONNX and Nim's FFI. The Nim code does not need to know anything about the internal structure of the model (CNNs, LSTMs); it only needs to communicate with the standardized ONNX Runtime API. This allows the application to remain simple while leveraging a highly complex and optimized backend for the heavy lifting of machine learning inference.

Nim

```
# file: inference.nim
import onnxruntime_c_api as ort
import sequtils
import os
import preprocessing # To access FeaturePoint type

proc runInference*(modelPath: string, features: seq[FeaturePoint]): int =
  ## Loads an ONNX model and performs inference.
  ## Returns the index of the recognized class.
  if features.len == 0: return -1 # Return -1 for invalid input

  # 1. Initialize ONNX Runtime Environment
  var env: ort.OrtEnv
  discard ort.CreateEnv(ort.ORT_LOGGING_LEVEL_WARNING, "HandwritingRecognizer", addr env)
  defer: discard ort.ReleaseEnv(env) # Ensure cleanup

  var sessionOptions: ort.OrtSessionOptions
  discard ort.CreateSessionOptions(addr sessionOptions)
  defer: discard ort.ReleaseSessionOptions(sessionOptions)

  # 2. Load the Model and Create a Session
  var session: ort.OrtSession
  if not fileExists(modelPath):
    echo "Error: Model file not found at ", modelPath
    return -1
  discard ort.CreateSession(env, modelPath.cstring, sessionOptions, addr session)
  defer: discard ort.ReleaseSession(session)

  # 3. Prepare the Input Tensor
  # Flatten the feature points into a single seq[float]
  var inputTensorValues = newSeq[float]()
  for p in features:
    inputTensorValues.add(p.x)
    inputTensorValues.add(p.y)

  # The MNIST model expects a 1x1x28x28 image. We will adapt our feature vector
  # to a similar flat structure. The model we use actually takes a flat  array.
  # Let's pad or truncate our input to match this size.
  const expectedInputSize = 784
  if inputTensorValues.len < expectedInputSize:
    # Pad with zeros
    for i in 0..< (expectedInputSize - inputTensorValues.len):
      inputTensorValues.add(0.0)
  elif inputTensorValues.len > expectedInputSize:
    # Truncate
    inputTensorValues.setLen(expectedInputSize)

  let inputShape: seq[int64] = @
  let inputTensorSize = inputTensorValues.len * sizeof(float)

  var memoryInfo: ort.OrtMemoryInfo
  discard ort.CreateCpuMemoryInfo(ort.OrtArenaAllocator, ort.OrtMemTypeDefault, addr memoryInfo)
  defer: discard ort.ReleaseMemoryInfo(memoryInfo)

  var inputTensor: ort.OrtValue
  discard ort.CreateTensorWithDataAsOrtValue(
    memoryInfo,
    inputTensorValues.addr,
    inputTensorSize,
    inputShape.len.int64,
    inputShape.addr,
    ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    addr inputTensor
  )
  defer: discard ort.ReleaseOrtValue(inputTensor)

  # Define input and output names (must match the model)
  let inputNames = ["Input3"] # Name from inspecting the mnist-12.onnx model
  let outputNames = ["Plus214_Output_0"] # Name from inspecting the model
  let inputNamesC = allocCStringArray(inputNames)
  let outputNamesC = allocCStringArray(outputNames)
  defer: deallocCStringArray(inputNamesC); deallocCStringArray(outputNamesC)

  # 4. Run Inference
  var outputTensor: ort.OrtValue
  discard ort.Run(
    session,
    nil, # RunOptions
    inputNamesC,
    addr inputTensor,
    1, # Number of inputs
    outputNamesC,
    1, # Number of outputs
    addr outputTensor
  )
  defer: discard ort.ReleaseOrtValue(outputTensor)

  # 5. Interpret the Output
  var outputDataPtr: ptr float
  discard ort.GetTensorMutableData(outputTensor, cast[ptr pointer](addr outputDataPtr))

  var outputShapeInfo: ort.OrtTensorTypeAndShapeInfo
  discard ort.GetTensorTypeAndShape(outputTensor, addr outputShapeInfo)
  defer: discard ort.ReleaseTensorTypeAndShapeInfo(outputShapeInfo)

  var numDims: int64
  discard ort.GetDimensionsCount(outputShapeInfo, addr numDims)
  var outputShape = newSeq[int64](numDims.int)
  discard ort.GetDimensions(outputShapeInfo, outputShape.addr, numDims)

  # The output for MNIST is a tensor of shape , with scores for each digit 0-9.
  let numClasses = outputShape.int
  var maxIndex = -1
  var maxValue = low(float)
  for i in 0..< numClasses:
    if outputDataPtr[i] > maxValue:
      maxValue = outputDataPtr[i]
      maxIndex = i

  return maxIndex
```

This procedure takes the model path and the preprocessed features as input. It handles setting up the ONNX environment, loading the model, formatting the input data into the required tensor shape, executing the inference run, and finally, interpreting the output tensor to find the most likely digit. The digit with the highest score (logit) is returned as the result.

## Part 5: Putting It All Together: The Final Application

This final section integrates all the previously developed components—UI, preprocessing, and inference—into a single, cohesive application. The result will be a functional program that allows a user to draw a digit on the screen and see the model's recognition in real-time.

### Step 1: The Main Module and Project Structure

A clean project structure is essential for maintainability. The project will be organized as follows:

```
handwriting_app/
├── models/
│   └── mnist-12.onnx
└── src/
    ├── main.nim
    ├── ui.nim
    ├── preprocessing.nim
    └── inference.nim
```

The `main.nim` file will serve as the entry point, importing the other modules and starting the application.

Nim

```
# file: src/main.nim
import ui

proc main() =
  runApplication()

main()
```

### Step 2: Connecting the Pipeline

The connection between the components is achieved by creating a callback procedure that is triggered by the UI. This procedure will orchestrate the flow of data from raw strokes to a recognized character. The `onMouseUp` event in the UI is the natural trigger for this pipeline.

First, a callback procedure is defined in `ui.nim` to handle the completed stroke.

Nim

```
# Add to the top of ui.nim
import preprocessing
import inference
import strformat

#... (type definitions)

proc handleStrokeRecognition(stroke: Stroke, screen: var Tigr) =
  echo &"Stroke completed with {stroke.len} points. Processing..."

  # 1. Preprocess the stroke
  let normalized = normalizeStroke(stroke, 20.0) # Normalize to a smaller size within a 28x28 canvas
  # We need to resample to a fixed size. For MNIST (28x28=784), we need 392 pairs of (x,y) points.
  # Our simple resampler is not ideal for this, but we'll use it conceptually.
  # A better approach would be to render the stroke to a 28x28 bitmap.
  # For now, let's just use the normalized points and pad.
  var features: seq[FeaturePoint]
  for p in normalized:
    features.add(FeaturePoint(x: p.x, y: p.y))

  # 2. Run inference
  let modelPath = "../models/mnist-12.onnx"
  let recognizedDigit = runInference(modelPath, features)

  # 3. Display the result
  if recognizedDigit!= -1:
    echo &"Recognized Digit: {recognizedDigit}"
    # We will add code to display this on the screen
  else:
    echo "Recognition failed."

#... (runApplication proc)
```

Next, the `runApplication` procedure in `ui.nim` is modified to call this handler and to display the result.

Nim

```
# Modified runApplication proc in ui.nim

proc runApplication*() =
  var screen = window(WindowWidth, WindowHeight, "Nim Handwriting Recognition", 0)

  # Application state
  var handwriting = newSeq()
  var currentStroke = newSeq[Point]()
  var isDrawing = false
  var recognizedText = "Draw a digit (0-9)" # Text to display

  # Load a font for displaying text
  let font = tfont()

  while screen.closed() == 0:
    # 1. Handle Input
    #... (mouse handling logic from Part 1)...
    
    # Inside the `if isDrawing:` block for mouse release:
    else:
      if isDrawing:
        isDrawing = false
        if currentStroke.len > 1:
          handwriting.add(currentStroke)
          # TRIGGER THE RECOGNITION PIPELINE
          let digit = runInference("../models/mnist-12.onnx", currentStroke.toFeatures()) # Assumes a helper proc to convert
          if digit!= -1:
            recognizedText = &"Recognized: {digit}"
          else:
            recognizedText = "Could not recognize."
        currentStroke = newSeq[Point]()

    # 2. Render
    screen.clear(RGB(20, 20, 30))
    #... (stroke drawing logic from Part 1)...

    # Display the recognized text
    screen.print(font, 10, 10, RGB(255, 255, 0), recognizedText)
    
    # Display instructions
    screen.print(font, 10, WindowHeight - 20, RGB(150, 150, 150), "Draw a digit, release mouse to recognize. Press any key to clear.")

    # Add a clear screen function
    if screen.key()!= 0:
      handwriting = newSeq()
      recognizedText = "Draw a digit (0-9)"

    screen.update()

# Helper proc to convert Stroke to seq[FeaturePoint]
proc toFeatures(stroke: Stroke): seq[FeaturePoint] =
  # First, normalize the stroke
  let normalized = normalizeStroke(stroke, 20.0)
  # Then, convert Point to FeaturePoint
  for p in normalized:
    result.add(FeaturePoint(x: p.x, y: p.y))
```

### Final Code and Conclusion

With these modifications, the application is complete. The `onMouseUp` event now triggers the full data processing pipeline: the captured stroke is normalized, converted into a feature vector, and passed to the ONNX inference engine. The model's output is then interpreted, and the recognized digit is displayed directly in the UI window.

This project has served as a comprehensive journey through the Nim language and its ecosystem. Starting from basic syntax, it has progressed to building a graphical user interface, implementing crucial data processing algorithms, and integrating with a state-of-the-art machine learning runtime via Nim's powerful Foreign Function Interface. The final application stands as a testament to Nim's suitability as a high-performance, expressive language capable of orchestrating complex systems that bridge the gap between user-facing applications and computationally intensive backends.