import { CONFIG } from "./config.js";

function normalizeText(text) {
    return String(text ?? "")
        .normalize("NFKC")
        .toLowerCase()
        .replace(/[！？!?。、,.「」『』（）()\[\]【】{}<>＜＞：:；;]/g, "")
        .replace(/\s+/g, "")
        .trim();
}

function makeNgrams(text, size = 2) {
    const normalized = normalizeText(text);
    const chars = [...normalized];
    const result = new Set();

    if (chars.length === 0) {
        return result;
    }

    if (chars.length <= size) {
        result.add(chars.join(""));
        return result;
    }

    for (let i = 0; i <= chars.length - size; i++) {
        result.add(chars.slice(i, i + size).join(""));
    }

    return result;
}

function diceSimilarity(setA, setB) {
    if (setA.size === 0 || setB.size === 0) {
        return 0;
    }

    let intersection = 0;

    for (const value of setA) {
        if (setB.has(value)) {
            intersection++;
        }
    }

    return (2 * intersection) / (setA.size + setB.size);
}

function characterSimilarity(a, b) {
    const aa = [...normalizeText(a)];
    const bb = [...normalizeText(b)];

    if (!aa.length || !bb.length) {
        return 0;
    }

    const countA = new Map();
    const countB = new Map();

    for (const char of aa) {
        countA.set(char, (countA.get(char) || 0) + 1);
    }

    for (const char of bb) {
        countB.set(char, (countB.get(char) || 0) + 1);
    }

    let common = 0;

    for (const [char, count] of countA) {
        common += Math.min(count, countB.get(char) || 0);
    }

    return (2 * common) / (aa.length + bb.length);
}

function containmentScore(a, b) {
    const aa = normalizeText(a);
    const bb = normalizeText(b);

    if (!aa || !bb) {
        return 0;
    }

    if (aa === bb) {
        return 1;
    }

    if (aa.includes(bb) || bb.includes(aa)) {
        const shorter = Math.min(aa.length, bb.length);
        const longer = Math.max(aa.length, bb.length);

        if (longer === 0) {
            return 0;
        }

        return 0.72 + (shorter / longer) * 0.28;
    }

    return 0;
}

function calculateSimilarity(a, b) {
    const normalizedA = normalizeText(a);
    const normalizedB = normalizeText(b);

    if (!normalizedA || !normalizedB) {
        return 0;
    }

    if (normalizedA === normalizedB) {
        return 1;
    }

    const containment = containmentScore(normalizedA, normalizedB);

    const bigramA = makeNgrams(normalizedA, 2);
    const bigramB = makeNgrams(normalizedB, 2);

    const trigramA = makeNgrams(normalizedA, 3);
    const trigramB = makeNgrams(normalizedB, 3);

    const bigramScore = diceSimilarity(bigramA, bigramB);
    const trigramScore = diceSimilarity(trigramA, trigramB);
    const charScore = characterSimilarity(normalizedA, normalizedB);

    let score =
        bigramScore * 0.40 +
        trigramScore * 0.30 +
        charScore * 0.20 +
        containment * 0.10;

    if (containment > score) {
        score = Math.max(score, containment * 0.95);
    }

    return Math.min(1, score);
}

function getSearchThreshold(input) {
    const length = [...normalizeText(input)].length;

    if (length <= 5) {
        return 0.62;
    }

    if (length <= 10) {
        return 0.52;
    }

    return 0.46;
}

function searchTrainingData(input, dataset) {
    if (!Array.isArray(dataset) || dataset.length === 0) {
        return {
            answer: null,
            score: 0,
            item: null,
            results: []
        };
    }

    const normalizedInput = normalizeText(input);

    if (!normalizedInput) {
        return {
            answer: null,
            score: 0,
            item: null,
            results: []
        };
    }

    for (const item of dataset) {
        if (!item || typeof item.input !== "string") {
            continue;
        }

        if (normalizeText(item.input) === normalizedInput) {
            return {
                answer: typeof item.output === "string" ? item.output : null,
                score: 1,
                item,
                results: [
                    {
                        item,
                        score: 1
                    }
                ]
            };
        }
    }

    const scored = [];

    for (const item of dataset) {
        if (!item || typeof item.input !== "string") {
            continue;
        }

        const score = calculateSimilarity(input, item.input);

        scored.push({
            item,
            score
        });
    }

    scored.sort((a, b) => b.score - a.score);

    const best = scored[0];

    if (!best) {
        return {
            answer: null,
            score: 0,
            item: null,
            results: []
        };
    }

    const threshold = getSearchThreshold(input);

    if (
        best.score >= threshold &&
        best.item &&
        typeof best.item.output === "string" &&
        best.item.output.trim()
    ) {
        return {
            answer: best.item.output,
            score: best.score,
            item: best.item,
            results: scored.slice(0, 5)
        };
    }

    return {
        answer: null,
        score: best.score,
        item: best.item,
        results: scored.slice(0, 5)
    };
}

function softmax(logits, temperature = 1) {
    if (!Array.isArray(logits) || logits.length === 0) {
        return [];
    }

    const safeTemperature = Math.max(0.05, Number(temperature) || 1);

    let max = -Infinity;

    for (const value of logits) {
        if (value > max) {
            max = value;
        }
    }

    const values = new Array(logits.length);
    let sum = 0;

    for (let i = 0; i < logits.length; i++) {
        const value = Math.exp((logits[i] - max) / safeTemperature);
        values[i] = value;
        sum += value;
    }

    if (!Number.isFinite(sum) || sum <= 0) {
        return logits.map(() => 1 / logits.length);
    }

    for (let i = 0; i < values.length; i++) {
        values[i] /= sum;
    }

    return values;
}

function applyTopK(probabilities, topK) {
    if (!Array.isArray(probabilities) || probabilities.length === 0) {
        return probabilities;
    }

    const k = Math.max(
        1,
        Math.min(
            probabilities.length,
            Number.isFinite(Number(topK))
                ? Math.floor(Number(topK))
                : probabilities.length
        )
    );

    if (k >= probabilities.length) {
        return probabilities;
    }

    const indices = probabilities
        .map((value, index) => ({
            value,
            index
        }))
        .sort((a, b) => b.value - a.value);

    const allowed = new Set();

    for (let i = 0; i < k; i++) {
        allowed.add(indices[i].index);
    }

    const result = new Array(probabilities.length).fill(0);

    let sum = 0;

    for (let i = 0; i < probabilities.length; i++) {
        if (allowed.has(i)) {
            result[i] = probabilities[i];
            sum += probabilities[i];
        }
    }

    if (sum <= 0) {
        return probabilities;
    }

    for (let i = 0; i < result.length; i++) {
        result[i] /= sum;
    }

    return result;
}

function applyTopP(probabilities, topP) {
    if (!Array.isArray(probabilities) || probabilities.length === 0) {
        return probabilities;
    }

    const p = Math.max(
        0.05,
        Math.min(
            1,
            Number.isFinite(Number(topP))
                ? Number(topP)
                : 1
        )
    );

    if (p >= 0.999) {
        return probabilities;
    }

    const sorted = probabilities
        .map((value, index) => ({
            value,
            index
        }))
        .sort((a, b) => b.value - a.value);

    const selected = new Set();
    let cumulative = 0;

    for (const item of sorted) {
        selected.add(item.index);
        cumulative += item.value;

        if (cumulative >= p) {
            break;
        }
    }

    const result = new Array(probabilities.length).fill(0);

    let sum = 0;

    for (let i = 0; i < probabilities.length; i++) {
        if (selected.has(i)) {
            result[i] = probabilities[i];
            sum += probabilities[i];
        }
    }

    if (sum <= 0) {
        return probabilities;
    }

    for (let i = 0; i < result.length; i++) {
        result[i] /= sum;
    }

    return result;
}

function randomChoice(probabilities) {
    let random = Math.random();

    for (let i = 0; i < probabilities.length; i++) {
        random -= probabilities[i];

        if (random <= 0) {
            return i;
        }
    }

    return Math.max(0, probabilities.length - 1);
}

function cleanGeneratedText(text) {
    if (typeof text !== "string") {
        return "";
    }

    let result = text
        .replace(/\u0000/g, "")
        .replace(/\r/g, "")
        .trim();

    result = result.replace(/(.)\1{6,}/gu, "$1$1");

    result = result.replace(/[^\p{L}\p{N}\p{P}\p{S}\p{Zs}\n]/gu, "");

    result = result
        .replace(/\s{3,}/g, " ")
        .trim();

    return result;
}

function generationQuality(text) {
    if (!text) {
        return 0;
    }

    const chars = [...text];

    if (chars.length < 2) {
        return 0.1;
    }

    let repeated = 0;

    for (let i = 1; i < chars.length; i++) {
        if (chars[i] === chars[i - 1]) {
            repeated++;
        }
    }

    const repeatedRatio = repeated / Math.max(1, chars.length - 1);

    const unique = new Set(chars).size;
    const diversity = unique / chars.length;

    let quality = 0.65;

    quality -= repeatedRatio * 0.9;

    if (diversity < 0.25) {
        quality -= 0.3;
    }

    if (chars.length < 8) {
        quality -= 0.15;
    }

    return Math.max(0, Math.min(1, quality));
}

function getConfigValue(name, fallback) {
    if (
        typeof CONFIG !== "undefined" &&
        CONFIG &&
        CONFIG[name] !== undefined
    ) {
        return CONFIG[name];
    }

    return fallback;
}

export function findTrainingAnswer(input, dataset) {
    return searchTrainingData(input, dataset);
}

export function searchKnowledge(input, dataset, limit = 5) {
    const normalizedInput = normalizeText(input);

    if (!normalizedInput || !Array.isArray(dataset)) {
        return [];
    }

    const results = dataset
        .filter(item => item && typeof item.input === "string")
        .map(item => ({
            item,
            score: calculateSimilarity(input, item.input)
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, Math.max(1, limit));

    return results;
}

export function generateResponse(model, tokenizer, input, dataset = [], options = {}) {
    const searchResult = searchTrainingData(input, dataset);

    if (searchResult.answer) {
        return searchResult.answer;
    }

    if (!model || !tokenizer) {
        return "まだ学習した回答がありません。";
    }

    const temperature =
        Number(options.temperature) ||
        Number(getConfigValue("temperature", 0.7));

    const topK =
        Number(options.topK) ||
        Number(getConfigValue("topK", 12));

    const topP =
        Number(options.topP) ||
        Number(getConfigValue("topP", 0.9));

    const maxGeneration =
        Number(options.maxGeneration) ||
        Number(getConfigValue("maxGeneration", 80));

    const contextSize =
        Number(getConfigValue("contextSize", 12));

    const inputTokens = tokenizer.encode(input);

    if (!inputTokens.length) {
        return "質問を入力してください。";
    }

    const generated = [...inputTokens];

    for (let step = 0; step < maxGeneration; step++) {
        const context = generated.slice(-contextSize);

        const logits = model.forward(context);

        if (!Array.isArray(logits) || logits.length === 0) {
            break;
        }

        let probabilities = softmax(logits, temperature);

        probabilities = applyTopK(probabilities, topK);
        probabilities = applyTopP(probabilities, topP);

        const nextToken = randomChoice(probabilities);

        if (
            tokenizer.eosId !== undefined &&
            nextToken === tokenizer.eosId
        ) {
            break;
        }

        generated.push(nextToken);

        if (generated.length > inputTokens.length + maxGeneration) {
            break;
        }
    }

    let text = "";

    try {
        text = tokenizer.decode(generated.slice(inputTokens.length));
    } catch {
        text = "";
    }

    text = cleanGeneratedText(text);

    const quality = generationQuality(text);

    if (quality < 0.25 || text.length < 2) {
        const fallbackResults = searchKnowledge(input, dataset, 3);

        if (
            fallbackResults.length > 0 &&
            fallbackResults[0].score >= getSearchThreshold(input) * 0.8 &&
            fallbackResults[0].item &&
            typeof fallbackResults[0].item.output === "string"
        ) {
            return fallbackResults[0].item.output;
        }

        return "質問に近い学習データがまだありません。もう少し具体的に質問してみてください。";
    }

    return text;
}

export function getSearchResults(input, dataset, limit = 5) {
    return searchKnowledge(input, dataset, limit);
}

export {
    normalizeText,
    makeNgrams,
    calculateSimilarity
};
