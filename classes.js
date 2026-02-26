export function makeCallable(instance) {
  const fn = (x) => instance.call(x);
  return fn;
}


export function isTfTensor(x) {
  return (
    x != null &&
    typeof x === 'object' &&
    Array.isArray(x.shape) &&      // shape exists like TF.js
    typeof x.data === 'function' && // has async data()
    typeof x.dataSync === 'function' && // has sync accessor
    typeof x.dtype === 'string'    // has dtype string
  );
}


/**
 * Map a TF.js dtype string to its corresponding ORT dtype string + TypedArray constructor.
 * @param {string} tfDtype
 * @returns {{ ortDtype: string, TypedArray: any }}
 */
function resolveDtype(tfDtype) {
  const map = {
    float32: { ortDtype: 'float32',  TypedArray: Float32Array   },
    float64: { ortDtype: 'double',   TypedArray: Float64Array   },
    int32:   { ortDtype: 'int32',    TypedArray: Int32Array     },
    uint8:   { ortDtype: 'uint8',    TypedArray: Uint8Array     },
    uint16:  { ortDtype: 'uint16',   TypedArray: Uint16Array    },
    int16:   { ortDtype: 'int16',    TypedArray: Int16Array     },
  };

  const resolved = map[tfDtype];
  if (!resolved) throw new Error(`Unsupported TF.js dtype: ${tfDtype}`);
  return resolved;
}

/**
 * Convert a TF.js tensor to an ort.Tensor.
 * @param {tf.Tensor} tfTensor
 * @returns {Promise<ort.Tensor>}
 */
export function tfTensorToOrt(tfTensor) {
  const { ortDtype, TypedArray } = resolveDtype(tfTensor.dtype);
  const shape = tfTensor.shape;
  const data = tfTensor.dataSync();
  return new ort.Tensor(ortDtype, data, shape);
}


/**
 * Convert an ort.Tensor to a TF.js tensor.
 * @param {ort.Tensor} ortTensor
 * @returns {tf.Tensor}
 */
export function ortTensorToTf(ortTensor) {
  const { type: ortDtype, dims, data } = ortTensor;
  const dtypeMap = {
    float32: 'float32',
    double:  'float64',
    int32:   'int32',
    uint8:   'uint8',
    uint16:  'uint16',
    int16:   'int16',
  };

  const tfDtype = dtypeMap[ortDtype];
  if (!tfDtype) throw new Error(`Unsupported ORT dtype: ${ortDtype}`);
  const shape = Array.from(dims);
  return tf.tensor(data, shape, tfDtype);
}


export class PretrainedONNXModel {
  constructor(deviceId=null, inputNames=null, outputNames=null) {
    this.inputNames = inputNames;
    this.outputNames = outputNames;
    this.loaded = false;
    this.deviceId = deviceId;
    this.session = null;
    this.pretrainedModelPath = null;
  }

  static async fromFile(pretrainedModelPath, deviceId=null, inputNames=null, outputNames=null, load=false) {
    const instance = new PretrainedONNXModel(deviceId, inputNames, outputNames);
    instance.pretrainedModelPath = pretrainedModelPath;
    if (load) {
      await instance.load();
    }
    return instance;
  }

  get device() {
    return this.deviceId === null ? "cpu" : `cuda:${this.deviceId}`;
  }

  unload() {
    this.loaded = false;
    if (this.session) {
      // onnxruntime sessions typically don't have an explicit close; drop reference
      this.session = null;
    }
  }

  async load() {
    if (this.loaded) return;
    let ort;
    try {
      ort = require("onnxruntime-node");
    } catch (e) {
      // fallback to global ort (e.g., onnxruntime-web loaded via script tag)
      ort = globalThis.ort;
    }

    if (!ort || !ort.InferenceSession) {
      throw new Error(
        "onnxruntime not found. Install `onnxruntime-node` (Node) or load `onnxruntime-web` (browser)."
      );
    }

    const createOptions = {};

    if (this.deviceId === null) {
      createOptions.executionProviders = ["cpu"];
    } else {
      createOptions.executionProviders = ["cuda", "cpu"];
      createOptions.deviceId = this.deviceId;
    }

    try {
      this.session = await ort.InferenceSession.create(
        this.pretrainedModelPath,
        createOptions
      );
      this.loaded = true;
    } catch (err) {
      try {
        const altOpts = { providers: createOptions.executionProviders };
        this.session = await ort.InferenceSession.create(
          this.pretrainedModelPath,
          altOpts
        );
        this.loaded = true;
      } catch (err2) {
        const message = `Failed to create ONNX InferenceSession: ${err.message}; fallback error: ${err2.message}`;
        throw new Error(message);
      }
    }
  }

  async run(inputs={}, options={}) {
    const retry = options.retry !== undefined ? !!options.retry : true;
    if (!this.loaded) {
      await this.load();
    }

    try {
      const output = await this.session.run(inputs);
      return output;
    } catch (err) {
      if (retry) {
        try {
          this.unload();
          await this.load();
          return await this.run(inputs, { retry: false });
        } catch (innerErr) {
          throw innerErr;
        }
      }
      throw err;
    }
  }

  async call(...inputTensors) {
    let inputs = {}
    let isTensor = false
    for (let i = 0; i < this.inputNames.length; i++) {
      if (isTfTensor(inputTensors[i])){
        isTensor = true
        inputTensors[i] = tfTensorToOrt(inputTensors[i])
      }
      inputs[this.inputNames[i]] = inputTensors[i];
    }
    let output = await this.session.run(inputs);
    let outputs = []
    if(isTensor){
      for (let i = 0; i < this.outputNames.length; i++){
        outputs.push(ortTensorToTf(output[this.outputNames[i]]))
      } 
    }
    else{
      for (let i = 0; i < this.outputNames.length; i++){
        outputs.push(output[this.outputNames[i]])
      } 
    }
    if (outputs.length === 1){
      return outputs[0];
    }
    return outputs;
  }

  getInputMetadata() {
    return this.session.inputMetadata;
  }

  getOutputMetadata() {
    return this.session.outputMetadata;
  }
}


export class EmbeddingPipeline {
  constructor(spectrogram, embedder) {
    this.spectrogram = spectrogram;
    this.embedder = embedder;
  }

  async audioToSpectrograms(audio, batchSize=128, melBins=32) {
    const [b, t] = audio.shape;
    const nFrames = Math.ceil(t / 160 - 3); // 160 = 10ms * 16kHz
    const chunks = [];
    for (let i = 0; i < Math.min(b, batchSize); i += batchSize) {
      let batch = audio.slice([i, 0], [Math.min(batchSize, b - i), -1]);
      let spec = await this.spectrogram(batch);
      let mel = tf.squeeze(spec, 1).div(10).add(2);
      chunks.push(mel);
      batch.dispose();
      spec.dispose();
    }
    const spectrograms = tf.concat(chunks, 0);    
    chunks.forEach(c => c.dispose());
    return spectrograms;
  }

  /* GENERATED WITH REFERENCES & GUIDANCES */
  async spectrogramsToEmbeddings(
      spectrograms,
      batchSize = 128,
      embeddingDim = 96,
      windowSize = 76,
      windowStride = 8
  ){
    const [b, t, melBins] = spectrograms.shape;

    // Same formula as Python
    const nFrames = Math.floor((t - windowSize) / windowStride) + 1;

    // Pre-allocate flat buffer (mirrors np.empty)
    const embeddingsData = new Float32Array(b * nFrames * embeddingDim);

    // Add channel dimension once: [b, t, melBins, 1]
    const specsWithChannel = tf.expandDims(spectrograms, -1);

    const batch = []; // {i: spectrogram_idx, j: start_frame, window: tf.Tensor3D}

    const processBatch = async () => {
      if (batch.length === 0) return;

      // Stack windows → [current_batch_size, windowSize, melBins, 1]
      const windowTensors = batch.map(item => item.window);
      const windowBatch = tf.stack(windowTensors);

      // Call embedder (exactly like Python)
      const result = await this.embedder(windowBatch);

      // Efficient bulk copy using dataSync
      const resultData = result.dataSync();

      for (let x = 0; x < batch.length; x++) {
        const { i, j } = batch[x];
        const frameIdx = Math.floor(j / windowStride);

        const offset = (i * nFrames + frameIdx) * embeddingDim;
        const vecOffset = x * embeddingDim;

        for (let d = 0; d < embeddingDim; d++) {
          embeddingsData[offset + d] = resultData[vecOffset + d];
        }

        // Clean up window immediately
        batch[x].window.dispose();
      }

      windowBatch.dispose();
      result.dispose();
      batch.length = 0;
    };

    // Exact same double loop as Python, but with TF.js slicing + tidy
    for (let i = 0; i < b; i++) {
      for (let j = 0; j < t; j += windowStride) {
        if (j + windowSize > t) break;

        // Extract window [windowSize, melBins, 1] (tidy disposes the temporary [1, ...] slice)
        const window = specsWithChannel.slice([i, j, 0, 0], [1, windowSize, -1, -1]).squeeze([0]);
        batch.push({ i, j, window });
        if (batch.length >= batchSize) {
          await processBatch();
        }
      }
    }

    // Remaining windows
    await processBatch();

    specsWithChannel.dispose();

    return tf.tensor3d(embeddingsData, [b, nFrames, embeddingDim], 'float32');
  }

  /* GENERATED WITH REFERENCES & GUIDANCES */
  async call(
    audio,
    spectrogramBatchSize = 32,
    melBins = 32,
    embeddingBatchSize = 32,
    embeddingDim = 96,
    windowSize = 76,
    windowStride = 8,
    audioWindowSize = 17280,
    audioWindowStride = 1920,
  ){
    audio = audio.mul(32767.0);

    const [batchSize, totalTime] = audio.shape;
    const embeddingsList = [];
    const spectrogramsList = [];

    // Slide windows across time (same logic as Python)
    for (let i = 0; i < totalTime - audioWindowSize + 1; i += audioWindowStride) {
      // Extract chunk [batchSize, audioWindowSize]
      const audioChunk = audio.slice([0, i], [-1, audioWindowSize]);

      // Compute mel spectrograms for this chunk
      const spectrograms = await this.audioToSpectrograms(
        audioChunk,
        spectrogramBatchSize,
        melBins
      );

      // Compute embeddings for this chunk
      const embeddings = await this.spectrogramsToEmbeddings(
        spectrograms,
        embeddingBatchSize,
        embeddingDim,
        windowSize,
        windowStride
      );

      embeddingsList.push(embeddings);

      spectrograms.dispose();
      audioChunk.dispose();
    }
    audio.dispose();

    // Concatenate along the time (embedding frames) dimension
    let finalEmbeddings;
    if (embeddingsList.length === 0) {
      throw new Error(`Empty embeddingList.`);
    } else {
      finalEmbeddings = tf.concat(embeddingsList, 1);
      embeddingsList.forEach(e => e.dispose());
    }

    return finalEmbeddings;
  }
}


export class WakeWordPipeline{
  constructor(){}
  
  static async fromFiles(spectrogramONNXPath, embedderONNXPath, wakeWordONNXPath) {
    let instance = new WakeWordPipeline()

    let spectrogram = await PretrainedONNXModel.fromFile(spectrogramONNXPath, null, ["input"], ["output"], true);
    let embedder = await PretrainedONNXModel.fromFile(embedderONNXPath, null, ["input_1"], ["conv2d_19"], true);  
    let wakeword = await PretrainedONNXModel.fromFile(wakeWordONNXPath, null, ["input"], ["output"], true);

    instance.spectrogram = makeCallable(spectrogram);
    instance.embedder = makeCallable(embedder);
    instance.wakeword = makeCallable(wakeword);
    
    let embeddingPipeline = new EmbeddingPipeline(instance.spectrogram, instance.embedder);
    instance.embeddingPipeline = makeCallable(embeddingPipeline);

    return makeCallable(instance);
  }

  async call(waveform){
    let prep = await this.embeddingPipeline(waveform);
    let final = await this.wakeword(prep);
    prep.dispose();
    return final;
  }

}