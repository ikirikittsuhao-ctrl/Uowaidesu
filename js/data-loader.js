import { CONFIG } from "./config.js";

function validateData(data, filename) {

    if (!Array.isArray(data)) {

        throw new Error(
            `${filename} は配列ではありません。`
        );
    }

    return data.filter(item => {

        if (!item) {
            return false;
        }

        if (
            typeof item.input !==
            "string"
        ) {
            return false;
        }

        if (
            typeof item.output !==
            "string"
        ) {
            return false;
        }

        if (
            !item.input.trim() ||
            !item.output.trim()
        ) {
            return false;
        }

        return true;
    });
}

export async function loadTrainingData() {

    const allData = [];

    for (
        const file of CONFIG.dataFiles
    ) {

        try {

            const response =
                await fetch(
                    file,
                    {
                        cache: "no-store"
                    }
                );

            if (!response.ok) {

                throw new Error(
                    `${response.status} ${response.statusText}`
                );
            }

            const data =
                await response.json();

            const valid =
                validateData(
                    data,
                    file
                );

            allData.push(
                ...valid
            );

        } catch (error) {

            console.error(
                `学習データ読み込み失敗: ${file}`,
                error
            );

            throw new Error(
                `${file} の読み込みに失敗しました。`
            );
        }
    }

    return removeDuplicates(
        allData
    );
}

function removeDuplicates(data) {

    const map = new Map();

    for (const item of data) {

        const key =
            `${item.input}\u0000${item.output}`;

        if (!map.has(key)) {
            map.set(key, item);
        }
    }

    return Array.from(
        map.values()
    );
}

export function mergeTrainingData(
    baseData,
    customData
) {

    return removeDuplicates([
        ...baseData,
        ...(Array.isArray(customData)
            ? customData
            : [])
    ]);
}
