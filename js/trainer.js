import { CONFIG } from "./config.js";

function sleep() {
    return new Promise(resolve => {
        requestAnimationFrame(() => resolve());
    });
}

function shuffle(array) {

    const result = [...array];

    for (
        let i = result.length - 1;
        i > 0;
        i--
    ) {

        const j =
            Math.floor(
                Math.random() * (i + 1)
            );

        [
            result[i],
            result[j]
        ] = [
            result[j],
            result[i]
        ];
    }

    return result;
}

export async function trainModel(
    model,
    tokenizer,
    dataset,
    options = {}
) {

    const epochs =
        Number(
            options.epochs ??
            CONFIG.epochs
        );

    const onProgress =
        typeof options.onProgress === "function"
            ? options.onProgress
            : () => {};

    if (!dataset.length) {
        return {
            loss: 0,
            epochs: 0
        };
    }

    let finalLoss = 0;

    for (let epoch = 1; epoch <= epochs; epoch++) {

        const shuffled =
            shuffle(dataset);

        let totalLoss = 0;
        let count = 0;

        for (
            let itemIndex = 0;
            itemIndex < shuffled.length;
            itemIndex++
        ) {

            const item =
                shuffled[itemIndex];

            const inputTokens =
                tokenizer.encode(
                    item.input
                );

            const outputTokens =
                tokenizer.encode(
                    item.output
                );

            if (
                inputTokens.length === 0 ||
                outputTokens.length === 0
            ) {
                continue;
            }

            const sequence =
                [
                    ...inputTokens,
                    ...outputTokens
                ];

            for (
                let i = inputTokens.length;
                i < sequence.length;
                i++
            ) {

                const target =
                    sequence[i];

                const context =
                    sequence.slice(
                        Math.max(
                            0,
                            i - tokenizer.size
                        ),
                        i
                    );

                const loss =
                    model.trainExample(
                        context,
                        target
                    );

                if (Number.isFinite(loss)) {
                    totalLoss += loss;
                    count++;
                }
            }

            if (itemIndex % 3 === 0) {
                await sleep();
            }
        }

        finalLoss =
            count > 0
                ? totalLoss / count
                : 0;

        const progress =
            epoch / epochs;

        onProgress({
            epoch,
            epochs,
            progress,
            loss: finalLoss
        });

        await sleep();
    }

    return {
        loss: finalLoss,
        epochs
    };
}
