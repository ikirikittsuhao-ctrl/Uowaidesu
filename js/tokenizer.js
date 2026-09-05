export class Tokenizer {

    constructor() {
        this.PAD = "<PAD>";
        this.UNK = "<UNK>";
        this.EOS = "<EOS>";

        this.tokenToId = new Map();
        this.idToToken = [];

        this.addToken(this.PAD);
        this.addToken(this.UNK);
        this.addToken(this.EOS);
    }

    normalize(text) {
        return String(text ?? "")
            .normalize("NFKC")
            .replace(/\r\n/g, "\n")
            .trim();
    }

    addToken(token) {
        if (!this.tokenToId.has(token)) {
            const id = this.idToToken.length;

            this.tokenToId.set(token, id);
            this.idToToken.push(token);

            return id;
        }

        return this.tokenToId.get(token);
    }

    build(dataset) {

        this.tokenToId.clear();
        this.idToToken.length = 0;

        this.addToken(this.PAD);
        this.addToken(this.UNK);
        this.addToken(this.EOS);

        for (const item of dataset) {

            const input = this.normalize(item.input);
            const output = this.normalize(item.output);

            for (const char of input) {
                this.addToken(char);
            }

            for (const char of output) {
                this.addToken(char);
            }
        }

        return this;
    }

    encode(text) {

        const normalized = this.normalize(text);
        const result = [];

        for (const char of normalized) {

            if (this.tokenToId.has(char)) {
                result.push(this.tokenToId.get(char));
            } else {
                result.push(this.tokenToId.get(this.UNK));
            }
        }

        return result;
    }

    decode(ids) {

        let result = "";

        for (const id of ids) {

            if (
                id === this.tokenToId.get(this.EOS) ||
                id === this.tokenToId.get(this.PAD)
            ) {
                continue;
            }

            const token = this.idToToken[id];

            if (token) {
                result += token;
            }
        }

        return result;
    }

    get size() {
        return this.idToToken.length;
    }

    getId(token) {
        return this.tokenToId.get(token);
    }

    getToken(id) {
        return this.idToToken[id];
    }

    toJSON() {

        return {
            tokens: this.idToToken
        };
    }

    static fromJSON(data) {

        const tokenizer = new Tokenizer();

        tokenizer.tokenToId.clear();
        tokenizer.idToToken = Array.isArray(data.tokens)
            ? [...data.tokens]
            : [];

        tokenizer.idToToken.forEach((token, index) => {
            tokenizer.tokenToId.set(token, index);
        });

        return tokenizer;
    }
}
