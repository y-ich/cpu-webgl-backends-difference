function softmax(arr) {
    const alpha = Math.max(...arr);
    const exps = arr.map(e => Math.exp(e - alpha));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(e => e / sum);
}

async function testPrediction(backend) {
    const size = 7;
    const batches = 1;
    const boardWxH = size * size;
    const inputBufferChannels = 22;
    const inputGlobalBufferChannels = 14;
    const symmetriesBufferLength = 3;
    await tf.setBackend(backend);
    console.log("backend", tf.getBackend());
    const model = await tf.loadGraphModel("web_model/model.json");
    const bin_inputs = new Float32Array(batches * boardWxH * inputBufferChannels);
    const global_inputs = new Float32Array(batches * inputGlobalBufferChannels);
    const symmetries = new Array(symmetriesBufferLength);
    for (let i = 0; i < boardWxH; i++) {
        bin_inputs[0 * boardWxH * inputBufferChannels + i * inputBufferChannels] = 1.0;
    }
    global_inputs[0 * inputGlobalBufferChannels + 5] = -0.5;
    global_inputs[0 * inputGlobalBufferChannels + 6] = 1.0;
    global_inputs[0 * inputGlobalBufferChannels + 7] = 0.5;
    global_inputs[0 * inputGlobalBufferChannels + 7] = 1.0;
    global_inputs[0 * inputGlobalBufferChannels + 13] = -0.5;

    const results = await model.executeAsync({
        "swa_model/bin_inputs": tf.tensor(bin_inputs, [batches, boardWxH, inputBufferChannels], 'float32'),
        "swa_model/global_inputs": tf.tensor(global_inputs, [batches, inputGlobalBufferChannels], 'float32'),
        "swa_model/symmetries": tf.tensor(symmetries, [symmetriesBufferLength], 'bool'),
    });
    for (let i = 0; i < results.length; i++) {
        const result = results[i];
        const data = await result.data();
        if (result.size === (boardWxH + 1) * 2) {
            console.log(data);
            const policyRaw = [];
            for (let i = 0; i < boardWxH; i++) {
                policyRaw.push(data[i * 2]);
            }
            const policy = softmax(policyRaw);
            for (let y = 0; y < size; y++) {
                let line = `${(" " + (y + 1).toString())} `;
                for (let x = 0; x < size; x++) {
                    line += (policy[y * size + x] * 100).toFixed(0) + " ";
                }
                console.log(line);
            }
            console.log((policy[policy.length - 1] * 100).toFixed(0));
        }
    }
}

async function testPredictions() {
    for (backend of ["cpu", "webgl"]) {
        await testPrediction(backend);
    }
}

testPredictions();