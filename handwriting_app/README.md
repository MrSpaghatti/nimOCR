# Nim Online Handwriting Recognition Application

This application allows users to draw English text on a canvas, and it uses a pre-trained TrOCR (Transformer Optical Character Recognition) ONNX model to recognize the handwritten text in real-time. This project was initially based on the `tutorial.md` for digit recognition and has been updated to support full text recognition.

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
    *   **json:** For parsing model configuration files (usually included with Nim).
    *   **tables:** For vocabulary mapping (usually included with Nim).
    *   Install external libraries using Nimble:
        ```bash
        nimble install tigr onnxruntime
        ```

4.  **ONNX Runtime Shared Library:**
    *   The `onnxruntime` Nimble package is a wrapper around the ONNX Runtime C library. You need to have the actual shared library (`.dll` for Windows, `.so` for Linux, `.dylib` for macOS) available.
    *   Download it from the [ONNX Runtime GitHub Releases page](https://github.com/microsoft/onnxruntime/releases). Make sure to download the version appropriate for your operating system and CPU architecture.
    *   Extract the archive and place the shared library file (e.g., `onnxruntime.dll`, `libonnxruntime.so`) either:
        *   In the same directory where your compiled `main` executable will be (e.g., `handwriting_app/src/` if running `nim c -r main.nim` from within `src/`).
        *   In a directory that is part of your system's standard library search path.

5.  **Pre-trained TrOCR Model and Tokenizer Files:**
    *   This project uses the `Xenova/trocr-base-handwritten` model, which is an ONNX version of `microsoft/trocr-base-handwritten`.
    *   You need to download the following files from the [Xenova/trocr-base-handwritten Hugging Face repository](https://huggingface.co/Xenova/trocr-base-handwritten/tree/main):
        *   From the `onnx/` subfolder:
            *   `encoder_model.onnx`
            *   `decoder_model.onnx` (or `decoder_model_merged.onnx` - the UI currently expects `decoder_model.onnx`)
        *   From the main folder (root of the repository):
            *   `vocab.json`
            *   `config.json`
            *   `preprocessor_config.json`
            *   `generation_config.json` (if available, otherwise some defaults are used)
            *   `special_tokens_map.json` (good for reference, though IDs are often in other configs)
    *   Create a directory `handwriting_app/models/trocr/`.
    *   Place all the downloaded files into this `handwriting_app/models/trocr/` directory. The application expects them at `../models/trocr/` relative to the `src/` directory.

## Building and Running

1.  Ensure all prerequisites and model files are set up as described above.
2.  Navigate to the `handwriting_app/src` directory:
    ```bash
    cd handwriting_app/src
    ```
3.  Compile and run the main application:
    ```bash
    nim c -r main.nim
    ```
    Alternatively, to build an executable (e.g., named `main` or `main.exe` in the `src` directory):
    ```bash
    nim c main.nim
    ```
    Then run `./main` (or `main.exe` on Windows). If you build this way, ensure the `models/trocr/` directory path is correctly accessible from where you run the executable (e.g., run from `src/` or copy executable to project root and adjust paths if necessary).

## Common Errors and Solutions

*   **Error: `cannot open file: onnxruntime.dll` (or `libonnxruntime.so`, `libonnxruntime.dylib`)**
    *   **Cause:** The ONNX Runtime shared library is not found.
    *   **Solution:** See step 4 in Prerequisites. Ensure the library is in the executable's directory or system path.

*   **Error: `Error: Model file not found at ../models/trocr/...` or `Error: vocab.json not found...`**
    *   **Cause:** Model or tokenizer/config files are missing or in the wrong location.
    *   **Solution:** See step 5 in Prerequisites. Ensure all required files from `Xenova/trocr-base-handwritten` are downloaded and placed into `handwriting_app/models/trocr/`.

*   **Error: `Error creating ONNX Session...` or `Error during ONNX Run...`**
    *   **Cause 1:** Incorrect ONNX model files (e.g., corrupted download, wrong version).
    *   **Solution 1:** Re-download the ONNX files for encoder and decoder from `Xenova/trocr-base-handwritten`.
    *   **Cause 2 (Critical):** Mismatch between the ONNX model's expected input/output node names or tensor shapes and what's implemented in `inference.nim`. The current implementation uses:
        *   Encoder Input: `"pixel_values"` (shape `[1,3,384,384]`)
        *   Encoder Output: `"last_hidden_state"`
        *   Decoder Inputs: `"input_ids"`, `"attention_mask"`, `"encoder_hidden_states"`
        *   Decoder Output: `"logits"`
    *   **Solution 2:** If these are incorrect for the specific ONNX files you downloaded (especially if you chose a different decoder variant like one with `_past_`), you may need to inspect your `.onnx` files with a tool like Netron and update the node names in `inference.nim`.

*   **Compilation Error: `Error: cannot open file: tigr` or `Error: cannot open file: onnxruntime_c_api`**
    *   **Cause:** Required Nimble packages are not installed.
    *   **Solution:** Run `nimble install tigr onnxruntime`.

*   **Application runs, but recognition is poor or outputs gibberish:**
    *   **Cause 1:** Image preprocessing in `preprocessing.nim` might not perfectly align with the TrOCR model's training. The current `rasterizeStrokeToRGB` is basic. Line thickness, anti-aliasing, or exact pixel value distribution might matter.
    *   **Solution 1:** Experiment with `lineThickness` in `ui.nim` or refine the `rasterizeStrokeToRGB` and `prepareImageTensor` functions in `preprocessing.nim`. Check the TrOCR model's documentation for any very specific preprocessing nuances.
    *   **Cause 2:** Basic greedy decoding is used. For more complex handwriting, this might not be optimal.
    *   **Solution 2:** (Future enhancement) Implement beam search decoding in `inference.nim`.
    *   **Cause 3:** The detokenization in `inference.nim` is basic (handles `Ġ` for spaces). Full BPE detokenization can be more complex.
    *   **Solution 3:** (Future enhancement) Implement more robust detokenization if subword artifacts are common.

This README provides guidance for setting up and running the updated TrOCR-based handwriting recognition application.
