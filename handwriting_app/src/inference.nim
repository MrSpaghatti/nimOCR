import ऑनnxruntime_c_api as ort # Renaming to "onnxruntime_c_api" for clarity if not already aliased.
                             # The tutorial uses `onnxruntime_c_api as ort`. I'll ensure it's just `ort`.
import onnxruntime_c_api as ort_api # Using full name for direct calls if needed, `ort` for brevity
import sequtils
import os
import strutils # For `replace` if used in detokenization
import json # For parsing vocab.json and config files
import tables # For IdToTokenMap if using Table
# `preprocessing` import might not be needed directly by this file anymore if types are self-contained or passed.
# import preprocessing

# Type to hold tokenizer configuration and vocabulary
type
  DecoderConfig* = object
    decoderStartTokenId*: int64
    eosTokenId*: int64
    padTokenId*: int64
    maxLength*: int
    vocabSize*: int

  IdToTokenMap* = Table[int64, string] # Changed from seq[string]

proc loadTrOCTokenizerData*(
    vocabPath: string,
    generationConfigPath: string,
    mainConfigPath: string,
    tokenizerConfigPath: string
  ): (IdToTokenMap, DecoderConfig, bool) =
  ## Loads TrOCR tokenizer vocabulary and essential generation configuration.
  ## Returns (IdToTokenMap, DecoderConfig, success_flag).
  var idToToken: IdToTokenMap = initTable[int64, string]() # Initialize as Table
  var decConfig: DecoderConfig
  var success = true

  # Load vocab.json
  if not fileExists(vocabPath):
    echo "Error: vocab.json not found at ", vocabPath
    return (idToToken, decConfig, false)

  try:
    let vocabContent = readFile(vocabPath)
    let vocabJson = parseJson(vocabContent)

    for key, valueNode in vocabJson: # Iterating over JsonObject
      let tokenId = valueNode.getInt()
      idToToken[tokenId.int64] = key # Directly populate the table

  except JsonParsingError, JsonKindError:
    echo "Error parsing vocab.json: ", getCurrentExceptionMsg()
    return (idToToken, decConfig, false)

  if idToToken.len == 0: # Check table's length
    echo "Error: Vocabulary loaded as empty. Cannot proceed."
    return (idToToken, decConfig, false)

  # Load configuration files
  var genConfigJson: JsonNode
  if fileExists(generationConfigPath):
    try:
      genConfigJson = parseJson(readFile(generationConfigPath))
    except JsonParsingError, JsonKindError:
      echo "Error parsing generation_config.json: ", getCurrentExceptionMsg()
      # Continue to try main config if generation_config is faulty or missing keys

  var mainConfJson: JsonNode
  if fileExists(mainConfigPath):
     try:
      mainConfJson = parseJson(readFile(mainConfigPath))
     except JsonParsingError, JsonKindError:
      echo "Error parsing config.json: ", getCurrentExceptionMsg()
      # If all config files fail to load/parse, we might have issues.

  var tokenizerConfJson: JsonNode
  if fileExists(tokenizerConfigPath):
    try:
      tokenizerConfJson = parseJson(readFile(tokenizerConfigPath))
    except JsonParsingError, JsonKindError:
      echo "Error parsing tokenizer_config.json: ", getCurrentExceptionMsg()

  # Helper to safely get int64 config values with logging for missing keys
  proc getConfigValue(key: string, sources: seq[JsonNode], defaultVal: int64, isCritical: bool = false): int64 =
    for i, sourceNode in sources:
      if sourceNode != nil:
        # Check specific paths for some configs (e.g. decoder section in main config)
        if key == "decoder_start_token_id" or key == "eos_token_id" or key == "pad_token_id" or key == "vocab_size":
          if sourceNode.hasKey("decoder") and sourceNode["decoder"].hasKey(key):
            return sourceNode["decoder"][key].getInt(defaultVal)
        # General key check
        if sourceNode.hasKey(key):
          return sourceNode[key].getInt(defaultVal)
    if isCritical:
      echo &"CRITICAL Error: Config key '{key}' not found in any source. Using default: {defaultVal}."
      success = false # Mark overall loading as failed if a critical key is missing
    else:
      echo &"Warning: Config key '{key}' not found. Using default: {defaultVal}."
    return defaultVal

  # Helper to safely get int config values
  proc getConfigIntValue(key: string, sources: seq[JsonNode], defaultVal: int, isCritical: bool = false): int =
    for i, sourceNode in sources:
      if sourceNode != nil:
        if key == "max_length": # Special handling for max_length prioritization
            if i == 0 and sourceNode.hasKey("max_length"): # genConfigJson
                return sourceNode["max_length"].getInt(defaultVal)
            elif i == 1 and sourceNode.hasKey("decoder") and sourceNode["decoder"].hasKey("max_length"): # mainConfJson (decoder section)
                return sourceNode["decoder"]["max_length"].getInt(defaultVal)
            elif i == 2 and sourceNode.hasKey("model_max_length"): # tokenizerConfJson
                return sourceNode["model_max_length"].getInt(defaultVal)
        elif sourceNode.hasKey(key) : # General key check
             return sourceNode[key].getInt(defaultVal)

    if isCritical:
      echo &"CRITICAL Error: Config key '{key}' not found in any source. Using default: {defaultVal}."
      success = false
    else:
      echo &"Warning: Config key '{key}' not found. Using default: {defaultVal}."
    return defaultVal

  let configSources = @[genConfigJson, mainConfJson, tokenizerConfJson] # Order of priority for general keys

  decConfig.decoderStartTokenId = getConfigValue("decoder_start_token_id", configSources, 2, isCritical=true)
  decConfig.eosTokenId = getConfigValue("eos_token_id", configSources, 2, isCritical=true)
  decConfig.padTokenId = getConfigValue("pad_token_id", configSources, 1, isCritical=true)

  # Max length prioritization: generation_config.json -> config.json (decoder) -> tokenizer_config.json (model_max_length) -> default
  var maxLengthFound = false
  if genConfigJson != nil and genConfigJson.hasKey("max_length"):
    decConfig.maxLength = genConfigJson["max_length"].getInt(64)
    maxLengthFound = true
  elif mainConfJson != nil and mainConfJson.hasKey("decoder") and mainConfJson["decoder"].hasKey("max_length"):
    decConfig.maxLength = mainConfJson["decoder"]["max_length"].getInt(64)
    maxLengthFound = true
  elif tokenizerConfJson != nil and tokenizerConfJson.hasKey("model_max_length"):
    decConfig.maxLength = tokenizerConfJson["model_max_length"].getInt(64)
    maxLengthFound = true

  if not maxLengthFound:
    echo "Warning: 'max_length' or 'model_max_length' not found in config files. Using default: 64."
    decConfig.maxLength = 64

  decConfig.vocabSize = getConfigValue("vocab_size", configSources, 50265, isCritical=true).int

  if idToToken.len == 0: # This check is now after trying to load vocab
    success = false

  if decConfig.vocabSize != idToToken.len and idToToken.len > 0:
    echo &"Warning: Mismatch vocab_size from config ({decConfig.vocabSize}) and actual vocab len ({idToToken.len}). Using actual vocab length."
    decConfig.vocabSize = idToToken.len
  elif idToToken.len == 0 and decConfig.vocabSize > 0:
     echo "Error: Vocab is empty, but vocabSize in config is > 0. Tokenizer data inconsistent."
     success = false

  if not success:
      echo "Error: One or more critical tokenizer configurations could not be loaded."

  echo &"Tokenizer loaded. Vocab size: {idToToken.len}. DecoderStartID: {decConfig.decoderStartTokenId}, EOS_ID: {decConfig.eosTokenId}, MaxLength: {decConfig.maxLength}. Load success: {success}"
  return (idToToken, decConfig, success)

proc decodeTokenSequence*(ids: seq[int64], idToToken: IdToTokenMap, config: DecoderConfig): string =
  var tokens: seq[string]
  for id_num in ids:
    if id_num == config.decoderStartTokenId and tokens.len == 0: # Often, start token isn't part of the output string
      continue
    if id_num == config.eosTokenId:
      break
    if id_num == config.padTokenId:
      continue

    if idToToken.hasKey(id_num):
      tokens.add(idToToken[id_num])
    else:
      echo &"Warning: Unknown token ID during decoding: {id_num}"
      tokens.add("<unk>") # Or handle unknown better

  # Basic BPE detokenization: Replace "Ġ" with space, then join.
  # More complex BPE rules (from merges.txt) are not handled here.
  var decodedText = ""
  for i, token in tokens:
    var processedToken = token.replace("Ġ", " ")
    # Roberta tokenizer sometimes adds a prefix space to the very first token if it's not BOS.
    # This simple replacement handles the common case.
    # More sophisticated detokenizers handle this based on previous token context.
    if i == 0: processedToken = processedToken.strip(leading=true) # Avoid leading space if first actual token had Ġ
    decodedText.add(processedToken)

  return decodedText.strip() # Final trim

proc runInference*(modelPath: string, features: seq[float]): int =
  # This is the old MNIST runInference. It will be replaced by runTrOCRInference.
  # For now, let's keep it to avoid breaking main.nim immediately.
  # Or, we can comment it out and expect main.nim to be updated.
  # Let's comment it out to focus on TrOCR.
  discard """
  ## Loads an ONNX model and performs inference using a flat sequence of floats.
  ## Returns the index of the recognized class.
  ## Stub implementation based on tutorial.

  # The input `features` is now expected to be a flat seq[float] (e.g., 784 pixels).
  # The check in ui.nim `features.len == gridSize * gridSize` already ensures this.
  # However, a direct check here for expected size is also good.
  const expectedInputSize = 28 * 28 # For MNIST
  if features.len != expectedInputSize:
    echo &"Inference: Incorrect number of features. Expected {expectedInputSize}, got {features.len}."
    return -1

  echo "runInference called with ", features.len, " float features."
  echo "Attempting to use model: ", modelPath

  # Convert input features (seq[float] i.e. float64) to seq[float32]
  # as ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT corresponds to float32.
  var featuresF32 = newSeq[float32](features.len)
  for i, val in features:
    featuresF32[i] = val.float32

  # Use featuresF32 for ONNX tensor creation
  var inputTensorValues = featuresF32 # This is now seq[float32]

  # Placeholder: In a real implementation, this would contain the full logic
  # from Part 4, Step 2 of the tutorial, using `inputTensorValues` (which is seq[float32]).
  # Key changes from tutorial's ONNX section based on this new input:
  # - No need to flatten FeaturePoint.x, .y.
  # - The padding/truncation logic in the tutorial's ONNX example
  #   `if inputTensorValues.len < expectedInputSize:`
  #   `elif inputTensorValues.len > expectedInputSize:`
  #   is now less relevant if `ui.nim` guarantees `features.len == expectedInputSize`.
  #   However, keeping it could make `runInference` more robust if called from elsewhere.
  #   For now, assuming the check above (`features.len != expectedInputSize`) is sufficient.
  # 1. Initialize ONNX Runtime Environment
  # 2. Load the Model and Create a Session
  # 3. Prepare the Input Tensor
  # 4. Run Inference
  # 1. Initialize ONNX Runtime Environment
  var env: ptr ort.OrtEnv
  var status = ort.CreateEnv(ort.ORT_LOGGING_LEVEL_WARNING, "HandwritingRecognizer", addr env)
  if status != nil:
    echo "Error creating ONNX Env: ", ort.GetErrorMessage(status)
    # ort.ReleaseStatus(status) # Not available in onnxruntime_c_api.nim? Usually status is consumed.
    return -1
  defer: ort.ReleaseEnv(env)

  var sessionOptions: ptr ort.OrtSessionOptions
  status = ort.CreateSessionOptions(addr sessionOptions)
  if status != nil:
    echo "Error creating ONNX SessionOptions: ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseSessionOptions(sessionOptions)

  # 2. Load the Model and Create a Session
  if not fileExists(modelPath):
    echo "Error: Model file not found at ", modelPath
    return -1

  var session: ptr ort.OrtSession
  status = ort.CreateSession(env, modelPath.cstring, sessionOptions, addr session)
  if status != nil:
    echo "Error creating ONNX Session for model ", modelPath, ": ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseSession(session)

  # 3. Prepare the Input Tensor
  # `inputTensorValues` is already `features` (seq[float] of size 784)

  # Assuming Input Tensor Shape: [1, 784] (batch_size=1, 784 features)
  let inputShape: seq[int64] = @[1'i64, expectedInputSize.int64]
  # inputTensorValues is now seq[float32], so use sizeof(float32)
  let inputTensorSize = inputTensorValues.len * sizeof(float32)

  var memoryInfo: ptr ort.OrtMemoryInfo
  status = ort.CreateCpuMemoryInfo(ort.OrtArenaAllocator, ort.OrtMemTypeDefault, addr memoryInfo)
  if status != nil:
    echo "Error creating ONNX MemoryInfo: ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseMemoryInfo(memoryInfo)

  var inputTensor: ptr ort.OrtValue
  status = ort.CreateTensorWithDataAsOrtValue(
    memoryInfo,
    inputTensorValues.addr,
    inputTensorSize.uint64, # API expects uint64 for len
    inputShape[0].addr,     # ptr to shape data
    inputShape.len.uint64,  # number of dimensions in shape
    ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    addr inputTensor
  )
  if status != nil:
    echo "Error creating ONNX Input Tensor: ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseValue(inputTensor)

  # Define input and output names (must match the model)
  let inputNames = ["Input3"]
  let outputNames = ["Plus214_Output_0"]

  var inputNamesC = allocCStringArray(inputNames)
  var outputNamesC = allocCStringArray(outputNames)
  # Defer deallocation using a block to ensure it happens before other defers if that matters,
  # or simply at scope exit. Standard defer order is LIFO.
  defer:
    deallocCStringArray(inputNamesC)
    deallocCStringArray(outputNamesC)

  # 4. Run Inference
  var outputTensor: ptr ort.OrtValue
  status = ort.Run(
    session,
    nil, # RunOptions, can be nil for defaults
    inputNamesC,
    addr inputTensor, # Pass address of the OrtValue pointer
    1, # Number of inputs
    outputNamesC,
    1, # Number of outputs
    addr outputTensor
  )
  if status != nil:
    echo "Error during ONNX Run: ", ort.GetErrorMessage(status)
    if outputTensor != nil: ort.ReleaseValue(outputTensor) # Attempt to release if allocated
    return -1
  defer: ort.ReleaseValue(outputTensor)

  # 5. Interpret the Output
  var outputDataPtr: ptr float32 # Assuming model outputs float32
  status = ort.GetTensorMutableData(outputTensor, cast[ptr pointer](addr outputDataPtr))
  if status != nil:
    echo "Error getting ONNX Output Tensor Data: ", ort.GetErrorMessage(status)
    return -1

  var typeAndShapeInfo: ptr ort.OrtTensorTypeAndShapeInfo
  status = ort.GetTensorTypeAndShape(outputTensor, addr typeAndShapeInfo)
  if status != nil:
    echo "Error getting ONNX Output Tensor Type/Shape Info: ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseTensorTypeAndShapeInfo(typeAndShapeInfo)

  var numDims: uint64
  status = ort.GetDimensionsCount(typeAndShapeInfo, addr numDims)
  if status != nil: echo "Error GetDimensionsCount: ", ort.GetErrorMessage(status); return -1

  var outputShape = newSeq[int64](numDims)
  status = ort.GetDimensions(typeAndShapeInfo, outputShape.addr, numDims)
  if status != nil: echo "Error GetDimensions: ", ort.GetErrorMessage(status); return -1

  # The output for MNIST is typically a tensor of shape [batch_size, num_classes] e.g., [1, 10]
  # Let's assume num_classes is the last dimension.
  if outputShape.len == 0:
    echo "Error: Output tensor has 0 dimensions."
    return -1

  let numClasses = outputShape[outputShape.len-1].int # e.g., 10 for MNIST digits

  var maxIndex = -1
  var maxValue = low(float32) # Match outputDataPtr type

  # Check if outputDataPtr is not nil before dereferencing
  if outputDataPtr == nil:
    echo "Error: outputDataPtr is nil after GetTensorMutableData."
    return -1

  for i in 0..<numClasses:
    if outputDataPtr[i] > maxValue:
      maxValue = outputDataPtr[i]
      maxIndex = i

  echo &"Inference successful. Recognized digit index: {maxIndex}, Score: {maxValue}"
  return maxIndex
  """
  # return -1 # Placeholder for old function - completely removing it now
  # This function is now obsolete and will be replaced by runTrOCRInference
  # To avoid compilation errors if main still calls it, we can make it return a default
  # or raise an error. For now, let's make it clear it shouldn't be used.
  raise newException(ValueError, "runInference (for MNIST) is obsolete. Use runTrOCRInference.")

proc runTrOCRInference*(
    encoderModelPath: string,
    decoderModelPath: string,
    pixelValues: seq[float32], # Expect CHW float32 tensor data
    idToToken: IdToTokenMap,
    decConfig: DecoderConfig
  ): string =
  ## Performs inference using TrOCR encoder and decoder ONNX models.

  let ort = ort_api # Use `ort` as the alias for brevity in API calls

  # Basic input validation
  let expectedPixelCount = 3 * 384 * 384 # C*H*W
  if pixelValues.len != expectedPixelCount:
    echo &"Error: pixelValues length mismatch. Expected {expectedPixelCount}, got {pixelValues.len}"
    return "Error: Input pixel data size incorrect."

  var env: ptr ort.OrtEnv
  var status = ort.CreateEnv(ort.ORT_LOGGING_LEVEL_WARNING, "TrOCRInference", addr env)
  if status != nil:
    echo "Error creating ONNX Env: ", ort.GetErrorMessage(status)
    return "Error: ONNX Env creation failed."
  defer: ort.ReleaseEnv(env)

  var sessionOptions: ptr ort.OrtSessionOptions
  status = ort.CreateSessionOptions(addr sessionOptions)
  if status != nil:
    echo "Error creating ONNX SessionOptions: ", ort.GetErrorMessage(status)
    return "Error: ONNX SessionOptions creation failed."
  defer: ort.ReleaseSessionOptions(sessionOptions)

  # --- Encoder Pass ---
  if not fileExists(encoderModelPath):
    echo "Error: Encoder model file not found at ", encoderModelPath
    return "Error: Encoder model not found."

  var encoderSession: ptr ort.OrtSession
  status = ort.CreateSession(env, encoderModelPath.cstring, sessionOptions, addr encoderSession)
  if status != nil:
    echo "Error creating Encoder Session: ", ort.GetErrorMessage(status)
    return "Error: Encoder session creation failed."
  defer: ort.ReleaseSession(encoderSession)

  var memoryInfo: ptr ort.OrtMemoryInfo
  status = ort.CreateCpuMemoryInfo(ort.OrtArenaAllocator, ort.OrtMemTypeDefault, addr memoryInfo)
  if status != nil:
    echo "Error creating ONNX MemoryInfo: ", ort.GetErrorMessage(status)
    return "Error: MemoryInfo creation failed."
  defer: ort.ReleaseMemoryInfo(memoryInfo)

  # Prepare Encoder Input Tensor (pixel_values)
  let encoderInputShape: seq[int64] = @[1'i64, 3'i64, 384'i64, 384'i64] # batch, C, H, W
  let encoderInputTensorSize = pixelValues.len * sizeof(float32)
  var encoderInputTensor: ptr ort.OrtValue

  # Note: pixelValues is already seq[float32] as per new function signature
  status = ort.CreateTensorWithDataAsOrtValue(
    memoryInfo,
    cast[pointer](pixelValues.addr), # Direct cast to pointer
    encoderInputTensorSize.uint64,
    encoderInputShape[0].addr,
    encoderInputShape.len.uint64,
    ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, # float32
    addr encoderInputTensor
  )
  if status != nil:
    echo "Error creating Encoder Input Tensor: ", ort.GetErrorMessage(status)
    return "Error: Encoder input tensor creation failed."
  defer: ort.ReleaseValue(encoderInputTensor)

  let encoderInputNames = ["pixel_values"] # Confirmed by user
  let encoderOutputNames = ["last_hidden_state"] # Confirmed by user
  var encoderInputNamesC = allocCStringArray(encoderInputNames)
  var encoderOutputNamesC = allocCStringArray(encoderOutputNames)
  defer:
    deallocCStringArray(encoderInputNamesC)
    deallocCStringArray(encoderOutputNamesC)

  var encoderOutputTensor: ptr ort.OrtValue
  status = ort.Run(
    encoderSession, nil, encoderInputNamesC, addr encoderInputTensor, 1,
    encoderOutputNamesC, 1, addr encoderOutputTensor
  )
  if status != nil:
    echo "Error during Encoder Run: ", ort.GetErrorMessage(status)
    if encoderOutputTensor != nil: ort.ReleaseValue(encoderOutputTensor)
    return "Error: Encoder run failed."
  # Defer release of encoderOutputTensor until after decoder loop uses it.

  # (Decoder Loop and Detokenization to be implemented next)
  # For now, let's get the shape of encoderOutputTensor to verify
  var encOutTypeAndShapeInfo: ptr ort.OrtTensorTypeAndShapeInfo
  status = ort.GetTensorTypeAndShape(encoderOutputTensor, addr encOutTypeAndShapeInfo)
  if status != nil: echo "Error GetTensorTypeAndShape for encoder output: ", ort.GetErrorMessage(status); return "Error processing encoder output"
  defer: ort.ReleaseTensorTypeAndShapeInfo(encOutTypeAndShapeInfo)

  var encOutNumDims: uint64
  status = ort.GetDimensionsCount(encOutTypeAndShapeInfo, addr encOutNumDims)
  if status != nil: echo "Error GetDimensionsCount for encoder output: ", ort.GetErrorMessage(status); return "Error processing encoder output"

  var encOutShape = newSeq[int64](encOutNumDims)
  status = ort.GetDimensions(encOutTypeAndShapeInfo, encOutShape.addr, encOutNumDims)
  if status != nil: echo "Error GetDimensions for encoder output: ", ort.GetErrorMessage(status); return "Error processing encoder output"

  echo &"Encoder output shape: {encOutShape}"
  # Expected e.g. @[1, 256, 768] where 256 is num_patches ( (384/16)*(384/16) )

  # Defer release of encoderOutputTensor until the very end of the function,
  # as the decoder loop will need it.
  defer:
    if encoderOutputTensor != nil: ort.ReleaseValue(encoderOutputTensor)

  # --- Decoder Pass ---
  # Note on Decoder Model Choice:
  # This implementation assumes a decoder model variant (e.g., `decoder_model.onnx` from Xenova)
  # that can be called iteratively by passing the full sequence of generated `input_ids`
  # and does not require explicit management of `past_key_values` by the caller in the loop.
  # If using a decoder variant like `decoder_with_past_model.onnx`, the loop would need
  # to be adapted to handle `past_key_values` as inputs and outputs of the decoder.
  if not fileExists(decoderModelPath):
    echo "Error: Decoder model file not found at ", decoderModelPath
    return "Error: Decoder model not found."

  var decoderSession: ptr ort.OrtSession
  status = ort.CreateSession(env, decoderModelPath.cstring, sessionOptions, addr decoderSession)
  if status != nil:
    echo "Error creating Decoder Session: ", ort.GetErrorMessage(status)
    return "Error: Decoder session creation failed."
  defer: ort.ReleaseSession(decoderSession)

  var generatedTokenIds = newSeq[int64](0) # Initialize empty
  generatedTokenIds.add(decConfig.decoderStartTokenId)

  var allOutputLogits: seq[ptr ort.OrtValue] # To manage and release OrtValues from loop

  for stepNum in 0 ..< decConfig.maxLength:
    # Prepare input_ids tensor
    let currentInputIdsShape: seq[int64] = @[1'i64, generatedTokenIds.len.int64]
    echo &"  Decoder Step {stepNum}: input_ids shape: {currentInputIdsShape}"
    var currentInputIdsTensor: ptr ort.OrtValue
    status = ort.CreateTensorWithDataAsOrtValue(
      memoryInfo, cast[pointer](generatedTokenIds.addr), (generatedTokenIds.len * sizeof(int64)).uint64,
      currentInputIdsShape[0].addr, currentInputIdsShape.len.uint64,
      ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, addr currentInputIdsTensor
    )
    if status != nil: echo "Error creating input_ids tensor: ", ort.GetErrorMessage(status); return "Error in decoder loop"
    defer: ort.ReleaseValue(currentInputIdsTensor)

    # Prepare attention_mask tensor (all 1s)
    var attentionMaskData = newSeq[int64](generatedTokenIds.len)
    for i in 0 ..< attentionMaskData.len: attentionMaskData[i] = 1'i64
    let attentionMaskShape: seq[int64] = @[1'i64, attentionMaskData.len.int64]
    var attentionMaskTensor: ptr ort.OrtValue
    echo &"  Decoder Step {stepNum}: attention_mask shape: {attentionMaskShape}"
    status = ort.CreateTensorWithDataAsOrtValue(
      memoryInfo, cast[pointer](attentionMaskData.addr), (attentionMaskData.len * sizeof(int64)).uint64,
      attentionMaskShape[0].addr, attentionMaskShape.len.uint64,
      ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, addr attentionMaskTensor
    )
    if status != nil: echo "Error creating attention_mask tensor: ", ort.GetErrorMessage(status); return "Error in decoder loop"
    defer: ort.ReleaseValue(attentionMaskTensor)

    # Logging encoderOutputTensor shape (already logged once, but good to see it per step if needed, or ensure it's stable)
    # This was already logged after encoder pass: echo &"Encoder output shape: {encOutShape}"

    # Define decoder inputs for this step
    # Assuming decoder_model.onnx takes these standard names.
    # No explicit past_key_values handling for this version to keep it simpler,
    # relying on the model structure of decoder_model.onnx (not _with_past).
    # This means the full generated sequence is passed each time.
    let decoderInputTensors = [currentInputIdsTensor, attentionMaskTensor, encoderOutputTensor]
    var decoderInputTensorsPtr: seq[ptr ort.OrtValue] = @[]
    for tns in decoderInputTensors: decoderInputTensorsPtr.add(tns)

    # Node names should match user provided ones if they differ from these common ones
    let decoderInputNodeNames = ["input_ids", "attention_mask", "encoder_hidden_states"]
    let decoderOutputNodeNames = ["logits"]

    var decInputNamesC = allocCStringArray(decoderInputNodeNames)
    var decOutputNamesC = allocCStringArray(decoderOutputNodeNames)
    defer:
        deallocCStringArray(decInputNamesC)
        deallocCStringArray(decOutputNamesC)

    var currentOutputLogits: ptr ort.OrtValue
    status = ort.Run(
      decoderSession, nil, decInputNamesC, decoderInputTensorsPtr[0].addr, decoderInputNodeNames.len.uint64,
      decOutputNamesC, 1, addr currentOutputLogits
    )
    if status != nil:
      echo "Error during Decoder Run (step ", stepNum, "): ", ort.GetErrorMessage(status)
      if currentOutputLogits != nil: ort.ReleaseValue(currentOutputLogits)
      return "Error: Decoder run failed."
    allOutputLogits.add(currentOutputLogits) # Add for deferred release

    # Get logits data
    var logitsDataPtr: ptr float32
    status = ort.GetTensorMutableData(currentOutputLogits, cast[ptr pointer](addr logitsDataPtr))
    if status != nil || logitsDataPtr == nil:
      echo "Error getting Decoder Logits Data: ", if status != nil: ort.GetErrorMessage(status) else: "nil pointer"
      return "Error: Failed to get logits data."

    # Get shape of logits: expected [batch_size, seq_len, vocab_size]
    var logitsTypeShapeInfo: ptr ort.OrtTensorTypeAndShapeInfo
    status = ort.GetTensorTypeAndShape(currentOutputLogits, addr logitsTypeShapeInfo)
    if status != nil: echo "Error GetTensorTypeAndShape for logits: ", ort.GetErrorMessage(status); return "Error processing logits"
    defer: ort.ReleaseTensorTypeAndShapeInfo(logitsTypeShapeInfo)

    var logitsNumDims: uint64
    status = ort.GetDimensionsCount(logitsTypeShapeInfo, addr logitsNumDims)
    if status != nil: echo "Error GetDimensionsCount for logits: ", ort.GetErrorMessage(status); return "Error processing logits"

    var logitsShape = newSeq[int64](logitsNumDims)
    status = ort.GetDimensions(logitsTypeShapeInfo, logitsShape.addr, logitsNumDims)
    if status != nil: echo "Error GetDimensions for logits: ", ort.GetErrorMessage(status); return "Error processing logits"
    echo &"  Decoder Step {stepNum}: logits tensor shape: {logitsShape}"

    if logitsShape.len != 3 or logitsShape[0] != 1 or logitsShape[1] != generatedTokenIds.len.int64:
        echo &"Unexpected logits shape structure: {logitsShape}. Expected [1, {generatedTokenIds.len}, vocab_size]"
        return "Error: Logits shape structure mismatch."

    if logitsShape[2] != decConfig.vocabSize.int64:
        echo &"CRITICAL Error: Logits vocab_size dimension ({logitsShape[2]}) " &
             &"does not match configured vocab_size ({decConfig.vocabSize})."
        return "Error: Logits vocab dimension mismatch."

    # Perform argmax on the logits of the last token generated
    let lastTokenLogitsOffset = (generatedTokenIds.len - 1) * decConfig.vocabSize
    var nextTokenId: int64 = -1
    var maxLogitVal = low(float32)
    for k in 0 ..< decConfig.vocabSize:
      let currentLogit = logitsDataPtr[lastTokenLogitsOffset + k]
      if currentLogit > maxLogitVal:
        maxLogitVal = currentLogit
        nextTokenId = k.int64

    if nextTokenId == -1 :
        echo "Error: Argmax failed to find next token."
        break

    generatedTokenIds.add(nextTokenId)

    if nextTokenId == decConfig.eosTokenId:
      echo "EOS token generated. Stopping."
      break

    if stepNum == decConfig.maxLength - 1:
        echo "Max length reached."

  # Defer release of all output logit tensors from the loop
  defer:
    for val in allOutputLogits:
      if val != nil: ort.ReleaseValue(val)

  # Detokenize
  let recognizedText = decodeTokenSequence(generatedTokenIds, idToToken, decConfig)
  echo &"Raw generated IDs: {generatedTokenIds}"
  echo &"Detokenized text: '{recognizedText}'"

  return recognizedText

  # The input `features` is now expected to be a flat seq[float] (e.g., 784 pixels).
  # The check in ui.nim `features.len == gridSize * gridSize` already ensures this.
  # However, a direct check here for expected size is also good.
  const expectedInputSize = 28 * 28 # For MNIST
  if features.len != expectedInputSize:
    echo &"Inference: Incorrect number of features. Expected {expectedInputSize}, got {features.len}."
    return -1

  echo "runInference called with ", features.len, " float features."
  echo "Attempting to use model: ", modelPath

  # Convert input features (seq[float] i.e. float64) to seq[float32]
  # as ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT corresponds to float32.
  var featuresF32 = newSeq[float32](features.len)
  for i, val in features:
    featuresF32[i] = val.float32

  # Use featuresF32 for ONNX tensor creation
  var inputTensorValues = featuresF32 # This is now seq[float32]

  # Placeholder: In a real implementation, this would contain the full logic
  # from Part 4, Step 2 of the tutorial, using `inputTensorValues` (which is seq[float32]).
  # Key changes from tutorial's ONNX section based on this new input:
  # - No need to flatten FeaturePoint.x, .y.
  # - The padding/truncation logic in the tutorial's ONNX example
  #   `if inputTensorValues.len < expectedInputSize:`
  #   `elif inputTensorValues.len > expectedInputSize:`
  #   is now less relevant if `ui.nim` guarantees `features.len == expectedInputSize`.
  #   However, keeping it could make `runInference` more robust if called from elsewhere.
  #   For now, assuming the check above (`features.len != expectedInputSize`) is sufficient.
  # 1. Initialize ONNX Runtime Environment
  # 2. Load the Model and Create a Session
  # 3. Prepare the Input Tensor
  # 4. Run Inference
  # 1. Initialize ONNX Runtime Environment
  var env: ptr ort.OrtEnv
  var status = ort.CreateEnv(ort.ORT_LOGGING_LEVEL_WARNING, "HandwritingRecognizer", addr env)
  if status != nil:
    echo "Error creating ONNX Env: ", ort.GetErrorMessage(status)
    # ort.ReleaseStatus(status) # Not available in onnxruntime_c_api.nim? Usually status is consumed.
    return -1
  defer: ort.ReleaseEnv(env)

  var sessionOptions: ptr ort.OrtSessionOptions
  status = ort.CreateSessionOptions(addr sessionOptions)
  if status != nil:
    echo "Error creating ONNX SessionOptions: ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseSessionOptions(sessionOptions)

  # 2. Load the Model and Create a Session
  if not fileExists(modelPath):
    echo "Error: Model file not found at ", modelPath
    return -1

  var session: ptr ort.OrtSession
  status = ort.CreateSession(env, modelPath.cstring, sessionOptions, addr session)
  if status != nil:
    echo "Error creating ONNX Session for model ", modelPath, ": ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseSession(session)

  # 3. Prepare the Input Tensor
  # `inputTensorValues` is already `features` (seq[float] of size 784)

  # Assuming Input Tensor Shape: [1, 784] (batch_size=1, 784 features)
  let inputShape: seq[int64] = @[1'i64, expectedInputSize.int64]
  # inputTensorValues is now seq[float32], so use sizeof(float32)
  let inputTensorSize = inputTensorValues.len * sizeof(float32)

  var memoryInfo: ptr ort.OrtMemoryInfo
  status = ort.CreateCpuMemoryInfo(ort.OrtArenaAllocator, ort.OrtMemTypeDefault, addr memoryInfo)
  if status != nil:
    echo "Error creating ONNX MemoryInfo: ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseMemoryInfo(memoryInfo)

  var inputTensor: ptr ort.OrtValue
  status = ort.CreateTensorWithDataAsOrtValue(
    memoryInfo,
    inputTensorValues.addr,
    inputTensorSize.uint64, # API expects uint64 for len
    inputShape[0].addr,     # ptr to shape data
    inputShape.len.uint64,  # number of dimensions in shape
    ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    addr inputTensor
  )
  if status != nil:
    echo "Error creating ONNX Input Tensor: ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseValue(inputTensor)

  # Define input and output names (must match the model)
  let inputNames = ["Input3"]
  let outputNames = ["Plus214_Output_0"]

  var inputNamesC = allocCStringArray(inputNames)
  var outputNamesC = allocCStringArray(outputNames)
  # Defer deallocation using a block to ensure it happens before other defers if that matters,
  # or simply at scope exit. Standard defer order is LIFO.
  defer:
    deallocCStringArray(inputNamesC)
    deallocCStringArray(outputNamesC)

  # 4. Run Inference
  var outputTensor: ptr ort.OrtValue
  status = ort.Run(
    session,
    nil, # RunOptions, can be nil for defaults
    inputNamesC,
    addr inputTensor, # Pass address of the OrtValue pointer
    1, # Number of inputs
    outputNamesC,
    1, # Number of outputs
    addr outputTensor
  )
  if status != nil:
    echo "Error during ONNX Run: ", ort.GetErrorMessage(status)
    if outputTensor != nil: ort.ReleaseValue(outputTensor) # Attempt to release if allocated
    return -1
  defer: ort.ReleaseValue(outputTensor)

  # 5. Interpret the Output
  var outputDataPtr: ptr float32 # Assuming model outputs float32
  status = ort.GetTensorMutableData(outputTensor, cast[ptr pointer](addr outputDataPtr))
  if status != nil:
    echo "Error getting ONNX Output Tensor Data: ", ort.GetErrorMessage(status)
    return -1

  var typeAndShapeInfo: ptr ort.OrtTensorTypeAndShapeInfo
  status = ort.GetTensorTypeAndShape(outputTensor, addr typeAndShapeInfo)
  if status != nil:
    echo "Error getting ONNX Output Tensor Type/Shape Info: ", ort.GetErrorMessage(status)
    return -1
  defer: ort.ReleaseTensorTypeAndShapeInfo(typeAndShapeInfo)

  var numDims: uint64
  status = ort.GetDimensionsCount(typeAndShapeInfo, addr numDims)
  if status != nil: echo "Error GetDimensionsCount: ", ort.GetErrorMessage(status); return -1

  var outputShape = newSeq[int64](numDims)
  status = ort.GetDimensions(typeAndShapeInfo, outputShape.addr, numDims)
  if status != nil: echo "Error GetDimensions: ", ort.GetErrorMessage(status); return -1

  # The output for MNIST is typically a tensor of shape [batch_size, num_classes] e.g., [1, 10]
  # Let's assume num_classes is the last dimension.
  if outputShape.len == 0:
    echo "Error: Output tensor has 0 dimensions."
    return -1

  let numClasses = outputShape[outputShape.len-1].int # e.g., 10 for MNIST digits

  var maxIndex = -1
  var maxValue = low(float32) # Match outputDataPtr type

  # Check if outputDataPtr is not nil before dereferencing
  if outputDataPtr == nil:
    echo "Error: outputDataPtr is nil after GetTensorMutableData."
    return -1

  for i in 0..<numClasses:
    if outputDataPtr[i] > maxValue:
      maxValue = outputDataPtr[i]
      maxIndex = i

  echo &"Inference successful. Recognized digit index: {maxIndex}, Score: {maxValue}"
  return maxIndex
