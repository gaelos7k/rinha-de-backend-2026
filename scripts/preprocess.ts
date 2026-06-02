import { createReadStream, writeFileSync } from "node:fs";
import { createGunzip } from "node:zlib";

const inputPath = process.env.REF_INPUT_PATH ?? "resources/references.json.gz";
const vectorsPath = process.env.REF_VECTORS_PATH ?? "resources/references.bin";
const labelsPath = process.env.REF_LABELS_PATH ?? "resources/labels.bin";
const expectedCount = Number(process.env.REF_EXPECTED_COUNT ?? "3000000");
const vectorSize = 14;

if (!Number.isFinite(expectedCount) || expectedCount <= 0) {
    throw new Error("Invalid REF_EXPECTED_COUNT");
}

let vectors = new Uint8Array(expectedCount * vectorSize);
let labels = new Uint8Array(expectedCount);
let count = 0;

const toByte = (value: number): number => {
    if (!Number.isFinite(value)) {
        return 0;
    }

    if (value === -1) {
        return 255;
    }

    const clamped = Math.min(1, Math.max(0, value));
    return Math.round(clamped * 255);
};

const pushRecord = (record: { vector: number[]; label: string }) => {
    if (!Array.isArray(record.vector) || record.vector.length !== vectorSize) {
        throw new Error("Invalid vector shape");
    }

    if (count >= expectedCount) {
        throw new Error("Reference count exceeds expected");
    }

    const offset = count * vectorSize;
    for (let i = 0; i < vectorSize; i++) {
        const value = Number(record.vector[i]);
        vectors[offset + i] = toByte(value);
    }

    labels[count] = record.label === "fraud" ? 1 : 0;
    count += 1;
};

const decoder = new TextDecoder("utf-8");
let depth = 0;
let inString = false;
let escape = false;
let current = "";

const processText = (text: string) => {
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];

        if (depth === 0) {
            if (ch === "{") {
                depth = 1;
                current = "{";
                inString = false;
                escape = false;
            }
            continue;
        }

        current += ch;

        if (inString) {
            if (escape) {
                escape = false;
            } else if (ch === "\\") {
                escape = true;
            } else if (ch === "\"") {
                inString = false;
            }
            continue;
        }

        if (ch === "\"") {
            inString = true;
            continue;
        }

        if (ch === "{") {
            depth += 1;
            continue;
        }

        if (ch === "}") {
            depth -= 1;
            if (depth === 0) {
                const record = JSON.parse(current) as { vector: number[]; label: string };
                pushRecord(record);
                current = "";
            }
        }
    }
};

const stream = createReadStream(inputPath).pipe(createGunzip());

for await (const chunk of stream) {
    const text = decoder.decode(chunk, { stream: true });
    if (text.length > 0) {
        processText(text);
    }
}

const tail = decoder.decode();
if (tail.length > 0) {
    processText(tail);
}

if (depth !== 0) {
    throw new Error("Incomplete JSON object");
}

const finalVectors = vectors.subarray(0, count * vectorSize);
const finalLabels = labels.subarray(0, count);

writeFileSync(
    vectorsPath,
    Buffer.from(finalVectors.buffer, finalVectors.byteOffset, finalVectors.byteLength)
);
writeFileSync(
    labelsPath,
    Buffer.from(finalLabels.buffer, finalLabels.byteOffset, finalLabels.byteLength)
);

console.log(`Processed ${count} vectors`);
