import { CONFIG } from "./config.js";

function safeParse(value, fallback) {

    if (!value) {
        return fallback;
    }

    try {
        return JSON.parse(value);
    } catch {
        return fallback;
    }
}

export function saveModel(model) {

    try {

        localStorage.setItem(
            CONFIG.storageKey,
            JSON.stringify(
                model.toJSON()
            )
        );

        return true;

    } catch (error) {

        console.error(
            "モデル保存失敗:",
            error
        );

        return false;
    }
}

export function loadModel() {

    try {

        const raw =
            localStorage.getItem(
                CONFIG.storageKey
            );

        return safeParse(
            raw,
            null
        );

    } catch (error) {

        console.error(
            "モデル読み込み失敗:",
            error
        );

        return null;
    }
}

export function clearModel() {

    localStorage.removeItem(
        CONFIG.storageKey
    );
}

export function saveHistory(history) {

    try {

        localStorage.setItem(
            CONFIG.historyKey,
            JSON.stringify(history)
        );

        return true;

    } catch {
        return false;
    }
}

export function loadHistory() {

    try {

        return safeParse(
            localStorage.getItem(
                CONFIG.historyKey
            ),
            []
        );

    } catch {
        return [];
    }
}

export function clearHistory() {

    localStorage.removeItem(
        CONFIG.historyKey
    );
}

export function saveCustomData(data) {

    try {

        localStorage.setItem(
            CONFIG.dataKey,
            JSON.stringify(data)
        );

        return true;

    } catch {
        return false;
    }
}

export function loadCustomData() {

    try {

        return safeParse(
            localStorage.getItem(
                CONFIG.dataKey
            ),
            []
        );

    } catch {
        return [];
    }
}

export function clearCustomData() {

    localStorage.removeItem(
        CONFIG.dataKey
    );
}

export function downloadJSON(
    filename,
    data
) {

    const blob =
        new Blob(
            [
                JSON.stringify(
                    data,
                    null,
                    2
                )
            ],
            {
                type:
                    "application/json"
            }
        );

    const url =
        URL.createObjectURL(
            blob
        );

    const anchor =
        document.createElement(
            "a"
        );

    anchor.href = url;
    anchor.download = filename;

    document.body.appendChild(
        anchor
    );

    anchor.click();

    anchor.remove();

    setTimeout(
        () => URL.revokeObjectURL(url),
        1000
    );
}
