import { CONFIG } from "./config.js";

export class Trainer {
    constructor(model, tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;

        this.isTraining = false;
        this.stopRequested = false;

        this.lastLoss = 0;
        this.totalSteps = 0;
        this.totalEpochs = 0;
    }

    stop() {
        this.stopRequested = true;
    }

    normalizeExample(item) {
        if (!item) {
            return null;
        }

        if (
            typeof item.input !== "string" ||
            typeof item.output !== "string"
        ) {
            return null;
        }

        const input =
            item.input.trim();

        const output =
            item.output.trim();

        if (!input || !output) {
            return null;
        }

        return {
            input,
            output
        };
    }

    buildSequence(example) {
        const normalized =
            this.normalizeExample(example);

        if (!normalized) {
            return [];
        }

        const text =
            normalized.input +
            "\n" +
            normalized.output;

        return this.tokenizer.encode(text);
    }

    async train(
        dataset,
        epochs = CONFIG.epochs,
        onProgress = null
    ) {
        if (this.isTraining) {
            return {
                loss: this.lastLoss,
                steps: this.totalSteps
            };
        }

        if (
            !Array.isArray(dataset) ||
            dataset.length === 0
        ) {
            return {
                loss: 0,
                steps: 0
            };
        }

        this.isTraining = true;
        this.stopRequested = false;

        this.totalSteps = 0;
        this.lastLoss = 0;

        const validExamples =
            dataset
                .map(item =>
                    this.normalizeExample(item)
                )
                .filter(Boolean);

        if (validExamples.length === 0) {
            this.isTraining = false;

            return {
                loss: 0,
                steps: 0
            };
        }

        const learningRate =
            Number(CONFIG.learningRate) ||
            0.012;

        const safeEpochs =
            Math.max(
                1,
                Math.floor(
                    Number(epochs) || 1
                )
            );

        let totalLoss = 0;
        let totalCount = 0;

        for (
            let epoch = 0;
            epoch < safeEpochs;
            epoch++
        ) {
            if (this.stopRequested) {
                break;
            }

            const shuffled =
                this.shuffle(
                    validExamples
                );

            for (
                let index = 0;
                index < shuffled.length;
                index++
            ) {
                if (this.stopRequested) {
                    break;
                }

                const example =
                    shuffled[index];

                const sequence =
                    this.buildSequence(
                        example
                    );

                if (
                    sequence.length < 2
                ) {
                    continue;
                }

                for (
                    let i = 1;
                    i < sequence.length;
                    i++
                ) {
                    if (this.stopRequested) {
                        break;
                    }

                    const start =
                        Math.max(
                            0,
                            i -
                            Number(
                                CONFIG.contextSize
                            )
                        );

                    const context =
                        sequence.slice(
                            start,
                            i
                        );

                    const target =
                        sequence[i];

                    const loss =
                        this.model.trainExample(
                            context,
                            target,
                            learningRate
                        );

                    if (
                        Number.isFinite(loss)
                    ) {
                        totalLoss += loss;
                        totalCount++;
                    }

                    this.totalSteps++;

                    if (
                        onProgress &&
                        (
                            this.totalSteps % 25 === 0 ||
                            i === sequence.length - 1
                        )
                    ) {
                        const averageLoss =
                            totalCount > 0
                                ? totalLoss / totalCount
                                : 0;

                        this.lastLoss =
                            averageLoss;

                        onProgress({
                            epoch: epoch + 1,
                            epochs: safeEpochs,
                            step: this.totalSteps,
                            loss: averageLoss,
                            exampleIndex:
                                index + 1,
                            exampleCount:
                                shuffled.length,
                            progress:
                                (
                                    (
                                        epoch *
                                        shuffled.length +
                                        index + 1
                                    ) /
                                    (
                                        safeEpochs *
                                        shuffled.length
                                    )
                                ) * 100
                        });

                        await this.yieldToBrowser();
                    }
                }
            }

            this.totalEpochs++;

            if (onProgress) {
                onProgress({
                    epoch: epoch + 1,
                    epochs: safeEpochs,
                    step: this.totalSteps,
                    loss:
                        totalCount > 0
                            ? totalLoss / totalCount
                            : 0,
                    progress:
                        (
                            (epoch + 1) /
                            safeEpochs
                        ) * 100
                });
            }

            await this.yieldToBrowser();
        }

        this.lastLoss =
            totalCount > 0
                ? totalLoss / totalCount
                : 0;

        this.isTraining = false;

        return {
            loss: this.lastLoss,
            steps: this.totalSteps,
            epochs: this.totalEpochs,
            stopped:
                this.stopRequested
        };
    }

    shuffle(array) {
        const result =
            [...array];

        for (
            let i = result.length - 1;
            i > 0;
            i--
        ) {
            const j =
                Math.floor(
                    Math.random() *
                    (i + 1)
                );

            const temporary =
                result[i];

            result[i] =
                result[j];

            result[j] =
                temporary;
        }

        return result;
    }

    yieldToBrowser() {
        return new Promise(resolve => {
            setTimeout(
                resolve,
                0
            );
        });
    }
}

export async function trainModel(
    model,
    tokenizer,
    dataset,
    options = {}
) {
    const trainer =
        new Trainer(
            model,
            tokenizer
        );

    return trainer.train(
        dataset,
        options.epochs ||
            CONFIG.epochs,
        options.onProgress ||
            null
    );
}
