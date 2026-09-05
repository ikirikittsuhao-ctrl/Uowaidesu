import { CONFIG } from "./config.js";
import {
    sampleFromProbabilities
} from "./model.js";

function normalize(text) {

    return String(text ?? "")
        .normalize("NFKC")
        .trim()
        .toLowerCase();
}

function similarity(a, b) {

    const aa = normalize(a);
    const bb = normalize(b);

    if (!aa || !bb) {
        return 0;
    }

    if (aa === bb) {
        return 1;
    }

    const maxLength =
        Math.max(
            aa.length,
            bb.length
        );

    let same = 0;

    const minLength =
        Math.min(
            aa.length,
            bb.length
        );

    for (let i = 0; i < minLength; i++) {

        if (aa[i] === bb[i]) {
            same++;
        }
    }

    return same / maxLength;
}

function findTrainingAnswer(
    input,
    dataset
) {

    const normalized =
        normalize(input);

    for (const item of dataset) {

        if (
            normalize(item.input) ===
            normalized
        ) {
            return item.output;
        }
    }

    let best = null;
    let bestScore = 0;

    for (const item of dataset) {

        const score =
            similarity(
                input,
                item.input
            );

        if (score > bestScore) {
            bestScore = score;
            best = item;
        }
    }

    if (
        best &&
        bestScore >= 0.72
    ) {
        return best.output;
    }

    return null;
}

function removeRepeatedTail(text) {

    if (text.length < 8) {
        return text;
    }

    const half =
        Math.floor(
            text.length / 2
        );

    for (
        let length = 2;
        length <= half;
        length++
    ) {

        const a =
            text.slice(
                text.length - length * 2,
                text.length - length
            );

        const b =
            text.slice(
                text.length - length
            );

        if (
            a.length === length &&
            a === b
        ) {

            return text.slice(
                0,
                text.length - length
            );
        }
    }

    return text;
}

export function generateResponse(
    model,
    tokenizer,
    dataset,
    input,
    options = {}
) {

    const exact =
        findTrainingAnswer(
            input,
            dataset
        );

    if (exact) {
        return exact;
    }

    if (!model) {
        return "まだAIモデルが準備できていません。";
    }

    const maxGeneration =
        Number(
            options.maxGeneration ??
            CONFIG.maxGeneration
        );

    const temperature =
        Number(
            options.temperature ??
            CONFIG.temperature
        );

    const topK =
        Number(
            options.topK ??
            CONFIG.topK
        );

    const topP =
        Number(
            options.topP ??
            CONFIG.topP
        );

    const inputTokens =
        tokenizer.encode(input);

    if (!inputTokens.length) {
        return "もう少し詳しく入力してください。";
    }

    const generated = [];

    let context = [
        ...inputTokens
    ];

    const eosId =
        tokenizer.getId(
            tokenizer.EOS
        );

    for (
        let step = 0;
        step < maxGeneration;
        step++
    ) {

        const prediction =
            model.predict(
                context,
                temperature
            );

        const tokenId =
            sampleFromProbabilities(
                prediction.probabilities,
                topK,
                topP
            );

        if (
            tokenId === eosId
        ) {
            break;
        }

        generated.push(
            tokenId
        );

        context = [
            ...context,
            tokenId
        ].slice(
            -CONFIG.contextSize
        );

        if (
            generated.length >= 2
        ) {

            const text =
                tokenizer.decode(
                    generated
                );

            if (
                /[。！？!?]$/.test(text)
            ) {
                if (
                    text.length >= 8
                ) {
                    break;
                }
            }
        }
    }

    let result =
        tokenizer.decode(
            generated
        );

    result =
        removeRepeatedTail(
            result
        ).trim();

    if (!result) {

        const fallback =
            dataset[
                Math.floor(
                    Math.random() *
                    dataset.length
                )
            ];

        return fallback
            ? fallback.output
            : "うまく生成できませんでした。";
    }

    return result;
}
