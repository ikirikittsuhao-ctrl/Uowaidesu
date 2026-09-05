import { CONFIG } from "./config.js";

import { Tokenizer } from "./tokenizer.js";

import {
    TinyNeuralNetwork
} from "./model.js";

import {
    trainModel
} from "./trainer.js";

import {
    saveModel,
    loadModel,
    clearModel,
    saveHistory,
    loadHistory,
    clearHistory,
    saveCustomData,
    loadCustomData,
    downloadJSON
} from "./storage.js";

import {
    generateResponse
} from "./generator.js";

import {
    loadTrainingData,
    mergeTrainingData
} from "./data-loader.js";

const chat =
    document.getElementById(
        "chat"
    );

const messageInput =
    document.getElementById(
        "messageInput"
    );

const sendButton =
    document.getElementById(
        "sendButton"
    );

const trainButton =
    document.getElementById(
        "trainButton"
    );

const clearButton =
    document.getElementById(
        "clearButton"
    );

const addDataButton =
    document.getElementById(
        "addDataButton"
    );

const exportDataButton =
    document.getElementById(
        "exportDataButton"
    );

const importDataButton =
    document.getElementById(
        "importDataButton"
    );

const importDataInput =
    document.getElementById(
        "importDataInput"
    );

const settingsButton =
    document.getElementById(
        "settingsButton"
    );

const closeSettingsButton =
    document.getElementById(
        "closeSettingsButton"
    );

const sidePanel =
    document.getElementById(
        "sidePanel"
    );

const overlay =
    document.getElementById(
        "overlay"
    );

const typing =
    document.getElementById(
        "typing"
    );

const trainingModal =
    document.getElementById(
        "trainingModal"
    );

const trainingText =
    document.getElementById(
        "trainingText"
    );

const progressBar =
    document.getElementById(
        "progressBar"
    );

const trainingLoss =
    document.getElementById(
        "trainingLoss"
    );

const dataModal =
    document.getElementById(
        "dataModal"
    );

const closeDataModal =
    document.getElementById(
        "closeDataModal"
    );

const trainingInput =
    document.getElementById(
        "trainingInput"
    );

const trainingOutput =
    document.getElementById(
        "trainingOutput"
    );

const saveTrainingDataButton =
    document.getElementById(
        "saveTrainingDataButton"
    );

const resetModelButton =
    document.getElementById(
        "resetModelButton"
    );

const statusText =
    document.getElementById(
        "statusText"
    );

const modelStatus =
    document.getElementById(
        "modelStatus"
    );

const vocabSize =
    document.getElementById(
        "vocabSize"
    );

const dataSize =
    document.getElementById(
        "dataSize"
    );

const lossValue =
    document.getElementById(
        "lossValue"
    );

const temperature =
    document.getElementById(
        "temperature"
    );

const topK =
    document.getElementById(
        "topK"
    );

const topP =
    document.getElementById(
        "topP"
    );

const maxGeneration =
    document.getElementById(
        "maxGeneration"
    );

const epochs =
    document.getElementById(
        "epochs"
    );

const temperatureValue =
    document.getElementById(
        "temperatureValue"
    );

const topKValue =
    document.getElementById(
        "topKValue"
    );

const topPValue =
    document.getElementById(
        "topPValue"
    );

const maxGenerationValue =
    document.getElementById(
        "maxGenerationValue"
    );

const epochsValue =
    document.getElementById(
        "epochsValue"
    );

let baseTrainingData = [];

let customTrainingData = [];

let trainingData = [];

let tokenizer = null;

let model = null;

let chatHistory = [];

let isTraining = false;

let isGenerating = false;

function setStatus(text) {

    statusText.textContent =
        text;
}

function updateModelInfo() {

    modelStatus.textContent =
        model
            ? "準備完了"
            : "未学習";

    vocabSize.textContent =
        tokenizer
            ? tokenizer.size
            : 0;

    dataSize.textContent =
        trainingData.length;

    if (
        lossValue.textContent === ""
    ) {
        lossValue.textContent = "-";
    }
}

function openSettings() {

    sidePanel.classList.add(
        "open"
    );

    overlay.classList.add(
        "active"
    );
}

function closeSettings() {

    sidePanel.classList.remove(
        "open"
    );

    overlay.classList.remove(
        "active"
    );
}

settingsButton.addEventListener(
    "click",
    openSettings
);

closeSettingsButton.addEventListener(
    "click",
    closeSettings
);

overlay.addEventListener(
    "click",
    closeSettings
);

function updateSettingsUI() {

    temperatureValue.textContent =
        Number(
            temperature.value
        ).toFixed(2);

    topKValue.textContent =
        topK.value;

    topPValue.textContent =
        Number(
            topP.value
        ).toFixed(2);

    maxGenerationValue.textContent =
        maxGeneration.value;

    epochsValue.textContent =
        epochs.value;
}

[
    temperature,
    topK,
    topP,
    maxGeneration,
    epochs
].forEach(element => {

    element.addEventListener(
        "input",
        updateSettingsUI
    );

});

updateSettingsUI();

function addMessage(
    role,
    text
) {

    const wrapper =
        document.createElement(
            "div"
        );

    wrapper.className =
        `message ${role}`;

    const inner =
        document.createElement(
            "div"
        );

    const meta =
        document.createElement(
            "div"
        );

    meta.className =
        "message-meta";

    meta.textContent =
        role === "user"
            ? "あなた"
            : "Frontend AI";

    const content =
        document.createElement(
            "div"
        );

    content.className =
        "message-content";

    content.textContent =
        text;

    inner.appendChild(
        meta
    );

    inner.appendChild(
        content
    );

    wrapper.appendChild(
        inner
    );

    chat.appendChild(
        wrapper
    );

    chat.scrollTop =
        chat.scrollHeight;
}

function removeWelcome() {

    const welcome =
        document.getElementById(
            "welcome"
        );

    if (welcome) {
        welcome.remove();
    }
}

function restoreHistory() {

    if (!Array.isArray(chatHistory)) {
        chatHistory = [];
    }

    if (!chatHistory.length) {
        return;
    }

    removeWelcome();

    for (
        const item of chatHistory
    ) {

        if (
            !item ||
            !item.role ||
            !item.text
        ) {
            continue;
        }

        addMessage(
            item.role,
            item.text
        );
    }
}

function saveCurrentHistory() {

    saveHistory(
        chatHistory
    );
}

function setTyping(
    active
) {

    if (active) {
        typing.classList.add(
            "active"
        );
    } else {
        typing.classList.remove(
            "active"
        );
    }
}

function sleep(ms) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                ms
            )
    );
}

async function sendMessage(
    forcedText = null
) {

    if (
        isTraining ||
        isGenerating
    ) {
        return;
    }

    const text =
        forcedText !== null
            ? forcedText.trim()
            : messageInput.value.trim();

    if (!text) {
        return;
    }

    removeWelcome();

    if (
        forcedText === null
    ) {
        messageInput.value = "";
        autoResize();
    }

    addMessage(
        "user",
        text
    );

    chatHistory.push({
        role: "user",
        text
    });

    saveCurrentHistory();

    isGenerating = true;

    sendButton.disabled = true;

    setTyping(true);

    await sleep(120);

    try {

        if (!model) {

            const fallback =
                findFallbackAnswer(
                    text
                );

            addAssistantMessage(
                fallback
            );

            return;
        }

        const answer =
            generateResponse(
                model,
                tokenizer,
                trainingData,
                text,
                {
                    temperature:
                        Number(
                            temperature.value
                        ),

                    topK:
                        Number(
                            topK.value
                        ),

                    topP:
                        Number(
                            topP.value
                        ),

                    maxGeneration:
                        Number(
                            maxGeneration.value
                        )
                }
            );

        await typeAssistantMessage(
            answer
        );

    } catch (error) {

        console.error(
            error
        );

        addAssistantMessage(
            "回答の生成中にエラーが発生しました。"
        );
    } finally {

        isGenerating = false;

        sendButton.disabled = false;

        setTyping(false);

        messageInput.focus();
    }
}

function findFallbackAnswer(
    text
) {

    const normalized =
        text
            .normalize("NFKC")
            .trim();

    for (
        const item of trainingData
    ) {

        if (
            item.input
                .normalize("NFKC")
                .trim() ===
            normalized
        ) {

            return item.output;
        }
    }

    return "まだ十分に学習できていません。先に「学習」を実行してください。";
}

async function typeAssistantMessage(
    text
) {

    const wrapper =
        document.createElement(
            "div"
        );

    wrapper.className =
        "message assistant";

    const inner =
        document.createElement(
            "div"
        );

    const meta =
        document.createElement(
            "div"
        );

    meta.className =
        "message-meta";

    meta.textContent =
        "Frontend AI";

    const content =
        document.createElement(
            "div"
        );

    content.className =
        "message-content";

    content.textContent =
        "";

    inner.appendChild(
        meta
    );

    inner.appendChild(
        content
    );

    wrapper.appendChild(
        inner
    );

    chat.appendChild(
        wrapper
    );

    for (
        let i = 0;
        i < text.length;
        i++
    ) {

        content.textContent +=
            text[i];

        chat.scrollTop =
            chat.scrollHeight;

        if (
            i % 2 === 0
        ) {
            await sleep(8);
        }
    }

    chatHistory.push({
        role: "assistant",
        text
    });

    saveCurrentHistory();
}

function addAssistantMessage(
    text
) {

    addMessage(
        "assistant",
        text
    );

    chatHistory.push({
        role: "assistant",
        text
    });

    saveCurrentHistory();
}

sendButton.addEventListener(
    "click",
    () => sendMessage()
);

messageInput.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }
    }
);

messageInput.addEventListener(
    "input",
    autoResize
);

function autoResize() {

    messageInput.style.height =
        "auto";

    messageInput.style.height =
        Math.min(
            messageInput.scrollHeight,
            160
        ) + "px";
}

document
    .querySelectorAll(
        ".quick-button"
    )
    .forEach(button => {

        button.addEventListener(
            "click",
            () => {

                const prompt =
                    button.dataset.prompt;

                sendMessage(
                    prompt
                );
            }
        );
    });

clearButton.addEventListener(
    "click",
    () => {

        chatHistory = [];

        clearHistory();

        chat.innerHTML = `
            <div id="welcome" class="welcome">
                <div class="welcome-icon">✦</div>
                <h1>Frontend AI</h1>
                <p>
                    ブラウザだけで動作する小型AIです。
                    学習データをJSONから読み込み、
                    ブラウザ上で学習できます。
                </p>

                <div class="quick-grid">
                    <button
                        class="quick-button"
                        data-prompt="こんにちは"
                    >
                        こんにちは
                    </button>

                    <button
                        class="quick-button"
                        data-prompt="あなたの名前は？"
                    >
                        あなたの名前は？
                    </button>

                    <button
                        class="quick-button"
                        data-prompt="JavaScriptとは？"
                    >
                        JavaScriptとは？
                    </button>

                    <button
                        class="quick-button"
                        data-prompt="HTMLとは？"
                    >
                        HTMLとは？
                    </button>
                </div>
            </div>
        `;

        document
            .querySelectorAll(
                ".quick-button"
            )
            .forEach(button => {

                button.addEventListener(
                    "click",
                    () => {

                        sendMessage(
                            button.dataset.prompt
                        );
                    }
                );
            });
    }
);

function openDataModal() {

    dataModal.classList.add(
        "active"
    );
}

function closeDataModalWindow() {

    dataModal.classList.remove(
        "active"
    );

    trainingInput.value = "";
    trainingOutput.value = "";
}

addDataButton.addEventListener(
    "click",
    openDataModal
);

closeDataModal.addEventListener(
    "click",
    closeDataModalWindow
);

saveTrainingDataButton.addEventListener(
    "click",
    () => {

        const input =
            trainingInput.value.trim();

        const output =
            trainingOutput.value.trim();

        if (!input || !output) {

            alert(
                "入力と出力の両方を入力してください。"
            );

            return;
        }

        customTrainingData.push({
            input,
            output
        });

        saveCustomData(
            customTrainingData
        );

        trainingData =
            mergeTrainingData(
                baseTrainingData,
                customTrainingData
            );

        rebuildModel();

        updateModelInfo();

        closeDataModalWindow();

        alert(
            "学習データを追加しました。モデルを学習してください。"
        );
    }
);

function rebuildModel() {

    tokenizer =
        new Tokenizer();

    tokenizer.build(
        trainingData
    );

    model =
        new TinyNeuralNetwork(
            tokenizer.size
        );

    clearModel();

    lossValue.textContent =
        "-";

    updateModelInfo();
}

trainButton.addEventListener(
    "click",
    train
);

async function train() {

    if (isTraining) {
        return;
    }

    if (!trainingData.length) {

        alert(
            "学習データがありません。"
        );

        return;
    }

    isTraining = true;

    trainButton.disabled = true;

    trainingModal.classList.add(
        "active"
    );

    trainingText.textContent =
        "学習を開始しています...";

    progressBar.style.width =
        "0%";

    trainingLoss.textContent =
        "Loss: -";

    try {

        if (!tokenizer) {

            tokenizer =
                new Tokenizer();

            tokenizer.build(
                trainingData
            );
        }

        if (
            !model ||
            model.vocabSize !==
            tokenizer.size
        ) {

            model =
                new TinyNeuralNetwork(
                    tokenizer.size
                );
        }

        const result =
            await trainModel(
                model,
                tokenizer,
                trainingData,
                {
                    epochs:
                        Number(
                            epochs.value
                        ),

                    onProgress:
                        progress => {

                            const percent =
                                Math.round(
                                    progress.progress *
                                    100
                                );

                            progressBar.style.width =
                                `${percent}%`;

                            trainingText.textContent =
                                `Epoch ${progress.epoch} / ${progress.epochs}`;

                            trainingLoss.textContent =
                                `Loss: ${progress.loss.toFixed(5)}`;
                        }
                }
            );

        saveModel(
            model
        );

        lossValue.textContent =
            result.loss.toFixed(5);

        modelStatus.textContent =
            "学習済み";

        trainingText.textContent =
            "学習が完了しました。";

        progressBar.style.width =
            "100%";

        await sleep(500);

    } catch (error) {

        console.error(
            error
        );

        trainingText.textContent =
            "学習中にエラーが発生しました。";

        alert(
            error.message
        );

    } finally {

        trainingModal.classList.remove(
            "active"
        );

        trainButton.disabled =
            false;

        isTraining =
            false;

        updateModelInfo();
    }
}

resetModelButton.addEventListener(
    "click",
    () => {

        const confirmed =
            confirm(
                "学習済みモデルを削除して初期状態に戻しますか？"
            );

        if (!confirmed) {
            return;
        }

        clearModel();

        model =
            new TinyNeuralNetwork(
                tokenizer.size
            );

        lossValue.textContent =
            "-";

        modelStatus.textContent =
            "未学習";

        updateModelInfo();

        alert(
            "モデルをリセットしました。"
        );
    }
);

exportDataButton.addEventListener(
    "click",
    () => {

        downloadJSON(
            "frontend-ai-training-data.json",
            trainingData
        );
    }
);

importDataButton.addEventListener(
    "click",
    () => {

        importDataInput.click();
    }
);

importDataInput.addEventListener(
    "change",
    async event => {

        const file =
            event.target.files?.[0];

        if (!file) {
            return;
        }

        try {

            const text =
                await file.text();

            const data =
                JSON.parse(text);

            if (
                !Array.isArray(data)
            ) {

                throw new Error(
                    "JSONの形式が正しくありません。"
                );
            }

            const valid =
                data.filter(
                    item =>
                        item &&
                        typeof item.input ===
                            "string" &&
                        typeof item.output ===
                            "string" &&
                        item.input.trim() &&
                        item.output.trim()
                );

            customTrainingData =
                valid;

            saveCustomData(
                customTrainingData
            );

            trainingData =
                mergeTrainingData(
                    baseTrainingData,
                    customTrainingData
                );

            rebuildModel();

            updateModelInfo();

            alert(
                `${valid.length}件の学習データを読み込みました。`
            );

        } catch (error) {

            console.error(
                error
            );

            alert(
                `JSON読み込みエラー: ${error.message}`
            );

        } finally {

            importDataInput.value =
                "";
        }
    }
);

async function initialize() {

    try {

        setStatus(
            "学習データ読み込み中..."
        );

        baseTrainingData =
            await loadTrainingData();

        customTrainingData =
            loadCustomData();

        trainingData =
            mergeTrainingData(
                baseTrainingData,
                customTrainingData
            );

        tokenizer =
            new Tokenizer();

        tokenizer.build(
            trainingData
        );

        const savedModel =
            loadModel();

        if (
            savedModel &&
            savedModel.vocabSize ===
            tokenizer.size
        ) {

            model =
                TinyNeuralNetwork
                    .fromJSON(
                        savedModel
                    );

        } else {

            model =
                new TinyNeuralNetwork(
                    tokenizer.size
                );
        }

        chatHistory =
            loadHistory();

        restoreHistory();

        updateModelInfo();

        setStatus(
            "ブラウザAI準備完了"
        );

    } catch (error) {

        console.error(
            error
        );

        setStatus(
            "読み込みエラー"
        );

        modelStatus.textContent =
            "エラー";

        alert(
            `初期化に失敗しました。\n${error.message}\n\nJSONファイルを読み込むため、HTTPサーバー経由で起動してください。`
        );
    }
}

initialize();
