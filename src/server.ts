import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { FraudPayload, FraudScoreResponse, MccRiskMap, NormalizationConfig } from "./types";
import { vectorize } from "./vectorizer";
import { knnFraudScore } from "./knn";

const defaultNormalization: NormalizationConfig = {
    max_amount: 10000,
    max_installments: 12,
    amount_vs_avg_ratio: 10,
    max_minutes: 1440,
    max_km: 1000,
    max_tx_count_24h: 20,
    max_merchant_avg_amount: 10000
};

const safeResponse: FraudScoreResponse = {
    approved: true,
    fraud_score: 0
};

const readJson = <T>(path: string, fallback: T): T => {
    try {
        const raw = readFileSync(path, "utf-8");
        return JSON.parse(raw) as T;
    } catch {
        return fallback;
    }
};

const normalizationPath = fileURLToPath(new URL("../resources/normalization.json", import.meta.url));
const mccRiskPath = fileURLToPath(new URL("../resources/mcc_risk.json", import.meta.url));
const referencesPath = fileURLToPath(new URL("../resources/references.bin", import.meta.url));
const labelsPath = fileURLToPath(new URL("../resources/labels.bin", import.meta.url));

const normalization = readJson(normalizationPath, defaultNormalization);
const mccRisk = readJson<MccRiskMap>(mccRiskPath, {});

const dims = 14;
const k = 5;
const debugLoad = process.env.DEBUG_LOAD === "1";

const logLoad = (...args: unknown[]) => {
    if (debugLoad) console.error(...args);
};

const jsonResponse = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" }
    });

const main = () => {
    let references = new Uint8Array(0);
    let labels = new Uint8Array(0);

    try {
        const vectorsBuffer = readFileSync(referencesPath);
        const labelsBuffer = readFileSync(labelsPath);

        logLoad("loadReferences", {
            vectorsBytes: vectorsBuffer.byteLength,
            labelsBytes: labelsBuffer.byteLength
        });

        const vectorsView = new Uint8Array(
            vectorsBuffer.buffer,
            vectorsBuffer.byteOffset,
            vectorsBuffer.byteLength
        );
        const labelsView = new Uint8Array(
            labelsBuffer.buffer,
            labelsBuffer.byteOffset,
            labelsBuffer.byteLength
        );

        const vectorCount = Math.floor(vectorsView.length / dims);
        const usableCount = Math.min(vectorCount, labelsView.length);

        if (usableCount <= 0) {
            console.error("loadReferences: empty dataset, exiting");
            process.exit(1);
        }

        references = vectorsView.subarray(0, usableCount * dims);
        labels = labelsView.subarray(0, usableCount);

        logLoad("loadReferences ok", { usableCount });
    } catch (error) {
        console.error("loadReferences failed, exiting", error);
        process.exit(1);
    }

    const port = Number(process.env.PORT ?? "9999");

    Bun.serve({
        port,
        development: false,
        fetch: async (request) => {
            const url = new URL(request.url);

            if (url.pathname === "/ready") {
                return new Response("OK", { status: 200 });
            }

            if (url.pathname === "/fraud-score") {
                if (request.method !== "POST") {
                    return new Response(null, { status: 404 });
                }

                try {
                    const payload = (await request.json()) as FraudPayload;
                    const vector = vectorize(payload, normalization, mccRisk);
                    const score = knnFraudScore(vector, references, labels, k, dims);
                    const approved = score < 0.6;

                    return jsonResponse({ approved, fraud_score: score });
                } catch {
                    return jsonResponse(safeResponse);
                }
            }

            return new Response(null, { status: 404 });
        }
    });

    console.error("Server listening on port", port);
};

main();