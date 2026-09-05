export class TinyLanguageModel {
    constructor(vocabSize, options = {}) {
        this.vocabSize = Math.max(1, Number(vocabSize) || 1);

        this.embeddingSize = Math.max(
            8,
            Number(options.embeddingSize) || 48
        );

        this.hiddenSize = Math.max(
            16,
            Number(options.hiddenSize) || 128
        );

        this.learningRate = Number(options.learningRate) || 0.012;

        this.contextSize = Math.max(
            1,
            Number(options.contextSize) || 12
        );

        this.seed = Number(options.seed) || 123456789;

        this.randomState = this.seed;

        this.embedding = new Float32Array(
            this.vocabSize * this.embeddingSize
        );

        this.W1 = new Float32Array(
            this.embeddingSize * this.hiddenSize
        );

        this.b1 = new Float32Array(
            this.hiddenSize
        );

        this.W2 = new Float32Array(
            this.hiddenSize * this.vocabSize
        );

        this.b2 = new Float32Array(
            this.vocabSize
        );

        this.initialize();
    }

    random() {
        let value = this.randomState;

        value ^= value << 13;
        value ^= value >>> 17;
        value ^= value << 5;

        this.randomState = value;

        return (
            (value >>> 0) /
            4294967296
        );
    }

    randomNormal(scale = 1) {
        let u = 0;
        let v = 0;

        while (u === 0) {
            u = this.random();
        }

        while (v === 0) {
            v = this.random();
        }

        const z =
            Math.sqrt(-2 * Math.log(u)) *
            Math.cos(2 * Math.PI * v);

        return z * scale;
    }

    initialize() {
        const embeddingScale =
            1 / Math.sqrt(this.embeddingSize);

        const hiddenScale =
            Math.sqrt(2 / this.embeddingSize);

        const outputScale =
            1 / Math.sqrt(this.hiddenSize);

        for (let i = 0; i < this.embedding.length; i++) {
            this.embedding[i] =
                this.randomNormal(embeddingScale);
        }

        for (let i = 0; i < this.W1.length; i++) {
            this.W1[i] =
                this.randomNormal(hiddenScale);
        }

        for (let i = 0; i < this.W2.length; i++) {
            this.W2[i] =
                this.randomNormal(outputScale);
        }

        this.b1.fill(0);
        this.b2.fill(0);
    }

    relu(value) {
        return value > 0 ? value : 0;
    }

    reluDerivative(value) {
        return value > 0 ? 1 : 0;
    }

    getContextEmbedding(tokens) {
        const hidden = new Float32Array(
            this.embeddingSize
        );

        if (!tokens || tokens.length === 0) {
            return hidden;
        }

        const context = tokens.slice(
            -this.contextSize
        );

        for (const token of context) {
            const id =
                Math.max(
                    0,
                    Math.min(
                        this.vocabSize - 1,
                        Number(token) || 0
                    )
                );

            const offset =
                id * this.embeddingSize;

            for (
                let e = 0;
                e < this.embeddingSize;
                e++
            ) {
                hidden[e] +=
                    this.embedding[offset + e];
            }
        }

        const scale =
            1 / Math.sqrt(context.length);

        for (
            let e = 0;
            e < this.embeddingSize;
            e++
        ) {
            hidden[e] *= scale;
        }

        return hidden;
    }

    forward(tokens, returnCache = false) {
        const input =
            this.getContextEmbedding(tokens);

        const hiddenPre =
            new Float32Array(this.hiddenSize);

        const hidden =
            new Float32Array(this.hiddenSize);

        const logits =
            new Float32Array(this.vocabSize);

        for (
            let h = 0;
            h < this.hiddenSize;
            h++
        ) {
            let value = this.b1[h];

            for (
                let e = 0;
                e < this.embeddingSize;
                e++
            ) {
                value +=
                    input[e] *
                    this.W1[
                        e * this.hiddenSize + h
                    ];
            }

            hiddenPre[h] = value;
            hidden[h] = this.relu(value);
        }

        for (
            let v = 0;
            v < this.vocabSize;
            v++
        ) {
            let value = this.b2[v];

            for (
                let h = 0;
                h < this.hiddenSize;
                h++
            ) {
                value +=
                    hidden[h] *
                    this.W2[
                        h * this.vocabSize + v
                    ];
            }

            logits[v] = value;
        }

        if (!returnCache) {
            return Array.from(logits);
        }

        return {
            input,
            hiddenPre,
            hidden,
            logits
        };
    }

    softmax(logits) {
        let maxLogit = -Infinity;

        for (const value of logits) {
            if (value > maxLogit) {
                maxLogit = value;
            }
        }

        const probabilities =
            new Float32Array(logits.length);

        let sum = 0;

        for (
            let i = 0;
            i < logits.length;
            i++
        ) {
            const value =
                Math.exp(
                    logits[i] - maxLogit
                );

            probabilities[i] = value;
            sum += value;
        }

        if (!Number.isFinite(sum) || sum <= 0) {
            const uniform =
                1 / logits.length;

            probabilities.fill(uniform);

            return probabilities;
        }

        for (
            let i = 0;
            i < probabilities.length;
            i++
        ) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    trainExample(tokens, targetToken, learningRate = this.learningRate) {
        if (!Array.isArray(tokens) || tokens.length === 0) {
            return 0;
        }

        const target =
            Math.max(
                0,
                Math.min(
                    this.vocabSize - 1,
                    Number(targetToken) || 0
                )
            );

        const cache =
            this.forward(tokens, true);

        const probabilities =
            this.softmax(cache.logits);

        const targetProbability =
            Math.max(
                1e-9,
                probabilities[target]
            );

        const loss =
            -Math.log(targetProbability);

        const dLogits =
            new Float32Array(
                this.vocabSize
            );

        for (
            let v = 0;
            v < this.vocabSize;
            v++
        ) {
            dLogits[v] =
                probabilities[v];

            if (v === target) {
                dLogits[v] -= 1;
            }
        }

        const dHidden =
            new Float32Array(
                this.hiddenSize
            );

        for (
            let h = 0;
            h < this.hiddenSize;
            h++
        ) {
            let gradient = 0;

            for (
                let v = 0;
                v < this.vocabSize;
                v++
            ) {
                gradient +=
                    dLogits[v] *
                    this.W2[
                        h * this.vocabSize + v
                    ];
            }

            dHidden[h] =
                gradient *
                this.reluDerivative(
                    cache.hiddenPre[h]
                );
        }

        for (
            let h = 0;
            h < this.hiddenSize;
            h++
        ) {
            for (
                let v = 0;
                v < this.vocabSize;
                v++
            ) {
                const index =
                    h * this.vocabSize + v;

                this.W2[index] -=
                    learningRate *
                    cache.hidden[h] *
                    dLogits[v];
            }
        }

        for (
            let v = 0;
            v < this.vocabSize;
            v++
        ) {
            this.b2[v] -=
                learningRate *
                dLogits[v];
        }

        for (
            let e = 0;
            e < this.embeddingSize;
            e++
        ) {
            for (
                let h = 0;
                h < this.hiddenSize;
                h++
            ) {
                const gradient =
                    cache.input[e] *
                    dHidden[h];

                this.W1[
                    e * this.hiddenSize + h
                ] -=
                    learningRate *
                    gradient;
            }
        }

        for (
            let h = 0;
            h < this.hiddenSize;
            h++
        ) {
            this.b1[h] -=
                learningRate *
                dHidden[h];
        }

        const context =
            tokens.slice(-this.contextSize);

        const scale =
            1 / Math.sqrt(
                Math.max(1, context.length)
            );

        const embeddingGradient =
            new Float32Array(
                this.embeddingSize
            );

        for (
            let e = 0;
            e < this.embeddingSize;
            e++
        ) {
            let gradient = 0;

            for (
                let h = 0;
                h < this.hiddenSize;
                h++
            ) {
                gradient +=
                    this.W1[
                        e * this.hiddenSize + h
                    ] *
                    dHidden[h];
            }

            embeddingGradient[e] =
                gradient * scale;
        }

        for (const token of context) {
            const id =
                Math.max(
                    0,
                    Math.min(
                        this.vocabSize - 1,
                        Number(token) || 0
                    )
                );

            const offset =
                id * this.embeddingSize;

            for (
                let e = 0;
                e < this.embeddingSize;
                e++
            ) {
                this.embedding[offset + e] -=
                    learningRate *
                    embeddingGradient[e];
            }
        }

        return loss;
    }

    trainSequence(
        sequence,
        epochs = 1,
        learningRate = this.learningRate,
        onProgress = null
    ) {
        if (
            !Array.isArray(sequence) ||
            sequence.length < 2
        ) {
            return 0;
        }

        let totalLoss = 0;
        let steps = 0;

        for (
            let epoch = 0;
            epoch < epochs;
            epoch++
        ) {
            for (
                let i = 1;
                i < sequence.length;
                i++
            ) {
                const context =
                    sequence.slice(
                        Math.max(
                            0,
                            i - this.contextSize
                        ),
                        i
                    );

                const target =
                    sequence[i];

                totalLoss +=
                    this.trainExample(
                        context,
                        target,
                        learningRate
                    );

                steps++;

                if (
                    onProgress &&
                    steps % 100 === 0
                ) {
                    onProgress({
                        epoch,
                        steps,
                        loss:
                            totalLoss /
                            steps
                    });
                }
            }
        }

        return steps > 0
            ? totalLoss / steps
            : 0;
    }

    resizeVocabulary(newVocabSize) {
        newVocabSize =
            Math.max(
                1,
                Number(newVocabSize) || 1
            );

        if (
            newVocabSize === this.vocabSize
        ) {
            return;
        }

        const oldVocabSize =
            this.vocabSize;

        const oldEmbedding =
            this.embedding;

        const oldW2 =
            this.W2;

        const oldB2 =
            this.b2;

        this.vocabSize =
            newVocabSize;

        this.embedding =
            new Float32Array(
                this.vocabSize *
                this.embeddingSize
            );

        this.W2 =
            new Float32Array(
                this.hiddenSize *
                this.vocabSize
            );

        this.b2 =
            new Float32Array(
                this.vocabSize
            );

        const embeddingCopy =
            Math.min(
                oldVocabSize,
                this.vocabSize
            );

        for (
            let token = 0;
            token < embeddingCopy;
            token++
        ) {
            for (
                let e = 0;
                e < this.embeddingSize;
                e++
            ) {
                this.embedding[
                    token * this.embeddingSize + e
                ] =
                    oldEmbedding[
                        token * this.embeddingSize + e
                    ];
            }
        }

        for (
            let h = 0;
            h < this.hiddenSize;
            h++
        ) {
            const oldOffset =
                h * oldVocabSize;

            const newOffset =
                h * this.vocabSize;

            for (
                let v = 0;
                v < embeddingCopy;
                v++
            ) {
                this.W2[
                    newOffset + v
                ] =
                    oldW2[
                        oldOffset + v
                    ];
            }
        }

        for (
            let v = 0;
            v < embeddingCopy;
            v++
        ) {
            this.b2[v] =
                oldB2[v];
        }

        const embeddingScale =
            1 / Math.sqrt(
                this.embeddingSize
            );

        const outputScale =
            1 / Math.sqrt(
                this.hiddenSize
            );

        for (
            let i = oldVocabSize *
                this.embeddingSize;
            i < this.embedding.length;
            i++
        ) {
            this.embedding[i] =
                this.randomNormal(
                    embeddingScale
                );
        }

        for (
            let i = oldVocabSize *
                this.hiddenSize;
            i < this.W2.length;
            i++
        ) {
            this.W2[i] =
                this.randomNormal(
                    outputScale
                );
        }
    }

    toJSON() {
        return {
            version: 2,
            vocabSize: this.vocabSize,
            embeddingSize: this.embeddingSize,
            hiddenSize: this.hiddenSize,
            learningRate: this.learningRate,
            contextSize: this.contextSize,
            seed: this.seed,
            randomState: this.randomState,
            embedding: Array.from(
                this.embedding
            ),
            W1: Array.from(this.W1),
            b1: Array.from(this.b1),
            W2: Array.from(this.W2),
            b2: Array.from(this.b2)
        };
    }

    static fromJSON(data) {
        const model =
            new TinyLanguageModel(
                data.vocabSize,
                {
                    embeddingSize:
                        data.embeddingSize,
                    hiddenSize:
                        data.hiddenSize,
                    learningRate:
                        data.learningRate,
                    contextSize:
                        data.contextSize,
                    seed:
                        data.seed
                }
            );

        if (Array.isArray(data.embedding)) {
            model.embedding.set(
                data.embedding.slice(
                    0,
                    model.embedding.length
                )
            );
        }

        if (Array.isArray(data.W1)) {
            model.W1.set(
                data.W1.slice(
                    0,
                    model.W1.length
                )
            );
        }

        if (Array.isArray(data.b1)) {
            model.b1.set(
                data.b1.slice(
                    0,
                    model.b1.length
                )
            );
        }

        if (Array.isArray(data.W2)) {
            model.W2.set(
                data.W2.slice(
                    0,
                    model.W2.length
                )
            );
        }

        if (Array.isArray(data.b2)) {
            model.b2.set(
                data.b2.slice(
                    0,
                    model.b2.length
                )
            );
        }

        if (
            Number.isFinite(
                Number(data.randomState)
            )
        ) {
            model.randomState =
                Number(data.randomState);
        }

        return model;
    }
            }
