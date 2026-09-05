export const CONFIG = {
    embeddingSize: 32,
    hiddenSize: 96,
    contextSize: 16,

    learningRate: 0.008,

    temperature: 0.65,
    topK: 8,
    topP: 0.9,

    maxGeneration: 80,

    epochs: 30,

    storageKey: "frontend-ai-v3-model",
    dataKey: "frontend-ai-v3-data",
    historyKey: "frontend-ai-v3-history",

    dataFiles: [
        "./data/training-01.json",
        "./data/training-02.json",
        "./data/training-03.json",
        "./data/training-04.json",
        "./data/training-05.json",
        "./data/training-06.json",
        "./data/training-07.json"
    ]
};
