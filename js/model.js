import { CONFIG } from "./config.js";

function randomUniform(scale) {
    return (Math.random() * 2 - 1) * scale;
}

function relu(x) {
    return x > 0 ? x : 0;
}

function reluDerivative(x) {
    return x > 0 ? 1 : 0;
}

export function softmax(logits, temperature = 1) {

    const temp = Math.max(0.01, temperature);

    let maxValue = -Infinity;

    for (const value of logits) {
        if (value > maxValue) {
            maxValue = value;
        }
    }

    const probabilities = new Array(logits.length);

    let sum = 0;

    for (let i = 0; i < logits.length; i++) {

        const value = Math.exp(
            (logits[i] - maxValue) / temp
        );

        probabilities[i] = value;
        sum += value;
    }

    if (!Number.isFinite(sum) || sum <= 0) {

        const uniform = 1 / logits.length;

        return logits.map(() => uniform);
    }

    for (let i = 0; i < probabilities.length; i++) {
        probabilities[i] /= sum;
    }

    return probabilities;
}

export function sampleFromProbabilities(
    probabilities,
    topK = 8,
    topP = 0.9
) {

    const indexes = probabilities
        .map((probability, index) => ({
            index,
            probability
        }))
        .sort(
            (a, b) =>
                b.probability - a.probability
        );

    const selected = indexes.slice(
        0,
        Math.max(1, Math.min(topK, indexes.length))
    );

    let cumulative = 0;
    const nucleus = [];

    for (const item of selected) {

        nucleus.push(item);
        cumulative += item.probability;

        if (cumulative >= topP) {
            break;
        }
    }

    let total = 0;

    for (const item of nucleus) {
        total += item.probability;
    }

    if (total <= 0) {
        return nucleus[0].index;
    }

    let random = Math.random() * total;

    for (const item of nucleus) {

        random -= item.probability;

        if (random <= 0) {
            return item.index;
        }
    }

    return nucleus[nucleus.length - 1].index;
}

export class TinyNeuralNetwork {

    constructor(vocabSize) {

        this.embeddingSize = CONFIG.embeddingSize;
        this.hiddenSize = CONFIG.hiddenSize;
        this.contextSize = CONFIG.contextSize;
        this.vocabSize = vocabSize;

        this.inputSize =
            this.embeddingSize *
            this.contextSize;

        this.embedding = new Float32Array(
            vocabSize * this.embeddingSize
        );

        this.W1 = new Float32Array(
            this.inputSize * this.hiddenSize
        );

        this.b1 = new Float32Array(
            this.hiddenSize
        );

        this.W2 = new Float32Array(
            this.hiddenSize * vocabSize
        );

        this.b2 = new Float32Array(
            vocabSize
        );

        this.initialize();
    }

    initialize() {

        const embeddingScale =
            Math.sqrt(2 / this.embeddingSize);

        const w1Scale =
            Math.sqrt(2 / this.inputSize);

        const w2Scale =
            Math.sqrt(2 / this.hiddenSize);

        for (let i = 0; i < this.embedding.length; i++) {
            this.embedding[i] =
                randomUniform(embeddingScale);
        }

        for (let i = 0; i < this.W1.length; i++) {
            this.W1[i] =
                randomUniform(w1Scale);
        }

        for (let i = 0; i < this.W2.length; i++) {
            this.W2[i] =
                randomUniform(w2Scale);
        }
    }

    getEmbedding(tokenId) {

        const start =
            tokenId * this.embeddingSize;

        return this.embedding.subarray(
            start,
            start + this.embeddingSize
        );
    }

    buildContext(tokenIds) {

        const context = new Int32Array(
            this.contextSize
        );

        const start =
            Math.max(
                0,
                tokenIds.length - this.contextSize
            );

        const available =
            tokenIds.length - start;

        const offset =
            this.contextSize - available;

        for (let i = 0; i < available; i++) {

            context[offset + i] =
                tokenIds[start + i];
        }

        return context;
    }

    forward(tokenIds) {

        const context =
            this.buildContext(tokenIds);

        const input =
            new Float32Array(this.inputSize);

        for (let position = 0; position < this.contextSize; position++) {

            const tokenId = context[position];

            const embeddingStart =
                tokenId * this.embeddingSize;

            const inputStart =
                position * this.embeddingSize;

            for (let j = 0; j < this.embeddingSize; j++) {

                input[inputStart + j] =
                    this.embedding[
                        embeddingStart + j
                    ];
            }
        }

        const hidden =
            new Float32Array(this.hiddenSize);

        for (let h = 0; h < this.hiddenSize; h++) {

            let sum = this.b1[h];

            for (let i = 0; i < this.inputSize; i++) {

                sum +=
                    input[i] *
                    this.W1[
                        i * this.hiddenSize + h
                    ];
            }

            hidden[h] = relu(sum);
        }

        const logits =
            new Float32Array(this.vocabSize);

        for (let v = 0; v < this.vocabSize; v++) {

            let sum = this.b2[v];

            for (let h = 0; h < this.hiddenSize; h++) {

                sum +=
                    hidden[h] *
                    this.W2[
                        h * this.vocabSize + v
                    ];
            }

            logits[v] = sum;
        }

        return {
            context,
            input,
            hidden,
            logits
        };
    }

    predict(tokenIds, temperature = CONFIG.temperature) {

        const result =
            this.forward(tokenIds);

        return {
            ...result,
            probabilities:
                softmax(
                    result.logits,
                    temperature
                )
        };
    }

    trainExample(inputTokens, targetToken) {

        const result =
            this.forward(inputTokens);

        const {
            context,
            input,
            hidden,
            logits
        } = result;

        const probabilities =
            softmax(logits, 1);

        const target =
            Math.max(
                0,
                Math.min(
                    this.vocabSize - 1,
                    targetToken
                )
            );

        const targetProbability =
            Math.max(
                probabilities[target],
                1e-12
            );

        const loss =
            -Math.log(targetProbability);

        const dLogits =
            new Float32Array(
                this.vocabSize
            );

        for (let v = 0; v < this.vocabSize; v++) {
            dLogits[v] = probabilities[v];
        }

        dLogits[target] -= 1;

        const dHidden =
            new Float32Array(
                this.hiddenSize
            );

        for (let h = 0; h < this.hiddenSize; h++) {

            let sum = 0;

            for (let v = 0; v < this.vocabSize; v++) {

                sum +=
                    dLogits[v] *
                    this.W2[
                        h * this.vocabSize + v
                    ];
            }

            dHidden[h] =
                sum *
                reluDerivative(
                    hidden[h]
                );
        }

        const learningRate =
            CONFIG.learningRate;

        for (let h = 0; h < this.hiddenSize; h++) {

            for (let v = 0; v < this.vocabSize; v++) {

                const index =
                    h * this.vocabSize + v;

                this.W2[index] -=
                    learningRate *
                    dHidden[h] *
                    0;

                this.W2[index] -=
                    learningRate *
                    hidden[h] *
                    dLogits[v];
            }

            this.b2[h % this.vocabSize] -=
                learningRate *
                dLogits[h % this.vocabSize];
        }

        const dInput =
            new Float32Array(
                this.inputSize
            );

        for (let i = 0; i < this.inputSize; i++) {

            let sum = 0;

            for (let h = 0; h < this.hiddenSize; h++) {

                sum +=
                    dHidden[h] *
                    this.W1[
                        i * this.hiddenSize + h
                    ];
            }

            dInput[i] = sum;
        }

        for (let i = 0; i < this.inputSize; i++) {

            for (let h = 0; h < this.hiddenSize; h++) {

                this.W1[
                    i * this.hiddenSize + h
                ] -=
                    learningRate *
                    input[i] *
                    dHidden[h];
            }
        }

        for (let h = 0; h < this.hiddenSize; h++) {

            this.b1[h] -=
                learningRate *
                dHidden[h];
        }

        for (let position = 0; position < this.contextSize; position++) {

            const tokenId =
                context[position];

            const embeddingStart =
                tokenId *
                this.embeddingSize;

            const inputStart =
                position *
                this.embeddingSize;

            for (let j = 0; j < this.embeddingSize; j++) {

                this.embedding[
                    embeddingStart + j
                ] -=
                    learningRate *
                    dInput[
                        inputStart + j
                    ];
            }
        }

        return loss;
    }

    toJSON() {

        return {
            version: 3,
            embeddingSize: this.embeddingSize,
            hiddenSize: this.hiddenSize,
            contextSize: this.contextSize,
            vocabSize: this.vocabSize,

            embedding:
                Array.from(this.embedding),

            W1:
                Array.from(this.W1),

            b1:
                Array.from(this.b1),

            W2:
                Array.from(this.W2),

            b2:
                Array.from(this.b2)
        };
    }

    static fromJSON(data) {

        if (!data) {
            return null;
        }

        const model =
            new TinyNeuralNetwork(
                data.vocabSize
            );

        if (
            data.embeddingSize !==
            model.embeddingSize ||
            data.hiddenSize !==
            model.hiddenSize ||
            data.contextSize !==
            model.contextSize
        ) {
            return null;
        }

        if (
            !Array.isArray(data.embedding) ||
            !Array.isArray(data.W1) ||
            !Array.isArray(data.b1) ||
            !Array.isArray(data.W2) ||
            !Array.isArray(data.b2)
        ) {
            return null;
        }

        model.embedding.set(
            data.embedding
        );

        model.W1.set(
            data.W1
        );

        model.b1.set(
            data.b1
        );

        model.W2.set(
            data.W2
        );

        model.b2.set(
            data.b2
        );

        return model;
    }
}
