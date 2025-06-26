# Nim Online Handwriting Recognition Application

This application allows users to draw digits (0-9) on a canvas, and it uses a pre-trained ONNX model to recognize the handwritten digits in real-time. This project is based on the tutorial found in `tutorial.md`.

## Prerequisites

Before you can build and run this application, you will need the following installed:

1.  **Nim Compiler:**
    *   Follow the official installation instructions at [https://nim-lang.org/install.html](https://nim-lang.org/install.html).
    *   Ensure `nim` is in your system's PATH.

2.  **Nimble Package Manager:**
    *   Nimble is usually installed alongside the Nim compiler.
    *   Ensure `nimble` is in your system's PATH.

3.  **Required Nim Libraries:**
    *   **tigr:** For the graphical user interface.
    *   **onnxruntime:** For running the ONNX model.
    *   Install them using Nimble:
        ```bash
        nimble install tigr onnxruntime
        ```

4.  **ONNX Runtime Shared Library:**
    *   The `onnxruntime` Nimble package is a wrapper around the ONNX Runtime C library. You need to have the actual shared library (`.dll` for Windows, `.so` for Linux, `.dylib` for macOS) available.
    *   Download it from the [ONNX Runtime GitHub Releases page](https://github.com/microsoft/onnxruntime/releases). Make sure to download the version appropriate for your operating system and CPU architecture.
    *   Extract the archive and place the shared library file (e.g., `onnxruntime.dll`, `libonnxruntime.so`) either:
        *   In the same directory where your compiled `main` executable will be (e.g., inside the `handwriting_app` directory after compilation, or in `handwriting_app/src` if running directly from source with `nim c -r src/main.nim`).
        *   In a directory that is part of your system's standard library search path (e.g., `/usr/lib` on Linux, or a directory added to `PATH` on Windows).

5.  **Pre-trained MNIST Model:**
    *   Download the `mnist-12.onnx` model file. A common source for such models is the [Hugging Face Model Hub](https://huggingface.co/models) or directly from ONNX model zoos. For this project, the tutorial refers to `mnist-12.onnx`.
    *   Place the `mnist-12.onnx` file into the `handwriting_app/models/` directory. The application expects it to be at `../models/mnist-12.onnx` relative to the source files, or `models/mnist-12.onnx` relative to the project root if running the compiled executable from the project root.

## Building and Running

1.  Navigate to the `handwriting_app/src` directory:
    ```bash
    cd handwriting_app/src
    ```
2.  Compile and run the main application:
    ```bash
    nim c -r main.nim
    ```
    Alternatively, to build an executable (e.g., named `main` or `main.exe` in the `src` directory):
    ```bash
    nim c main.nim
    ```
    Then run `./main` (or `main.exe` on Windows). If you build this way, ensure the `models` directory is correctly pathed or copy the executable to the project root (`handwriting_app/`) and run it from there.

## Common Errors and Solutions

*   **Error: `cannot open file: onnxruntime.dll` (or `libonnxruntime.so`, `libonnxruntime.dylib`)**
    *   **Cause:** The ONNX Runtime shared library is not found by the application.
    *   **Solution:**
        1.  Ensure you have downloaded the correct ONNX Runtime shared library for your OS.
        2.  Place the library file (e.g., `onnxruntime.dll`) in the same directory as your compiled executable (e.g., `handwriting_app/src/` if running `nim c -r main.nim`, or `handwriting_app/` if you compiled and moved the executable there).
        3.  Alternatively, add the directory containing the shared library to your system's `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux/macOS).

*   **Error: `Error: Model file not found at ../models/mnist-12.onnx`**
    *   **Cause:** The `mnist-12.onnx` model file is not in the expected location.
    *   **Solution:**
        1.  Make sure you have downloaded the `mnist-12.onnx` file.
        2.  Place it inside the `handwriting_app/models/` directory.
        3.  The path used in `inference.nim` is relative. If running `nim c -r src/main.nim` from the `handwriting_app` directory, the path `models/mnist-12.onnx` might be more appropriate. If running from `handwriting_app/src`, then `../models/mnist-12.onnx` is correct. The boilerplate will use `../models/mnist-12.onnx` assuming execution from the `src` directory or that the executable is run from a location where this relative path is valid.

*   **Compilation Error: `Error: cannot open file: tigr` or `Error: cannot open file: onnxruntime_c_api`**
    *   **Cause:** The required Nimble packages (`tigr` or `onnxruntime`) are not installed.
    *   **Solution:** Run `nimble install tigr onnxruntime` in your terminal.

*   **Application runs, but recognition always fails or gives unexpected results.**
    *   **Cause 1:** The input preprocessing might not perfectly align with what the `mnist-12.onnx` model expects. The tutorial provides a conceptual preprocessing pipeline.
    *   **Solution 1:** Double-check the preprocessing steps, especially normalization and how features are fed to the model. The `mnist-12.onnx` model typically expects a 1x1x28x28 grayscale image or a flattened array of 784 pixels. The current boilerplate will set up stubs based on the tutorial, which might need refinement for optimal performance.
    *   **Cause 2:** The specific `mnist-12.onnx` model you downloaded might have slightly different input/output node names than what's in the tutorial's `inference.nim` example.
    *   **Solution 2:** Use a tool like Netron to inspect your `.onnx` model and verify the exact input and output node names. Update these names in `inference.nim` if they differ. The tutorial uses "Input3" and "Plus214_Output_0".

This README provides a starting point. Refer to `tutorial.md` for the full context and detailed explanations of each part of the application.
