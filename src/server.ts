import { readFileSync } from "node:fs";
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

const readJson = <T>(url: URL, fallback: T): T => {
    try {
        const raw = readFileSync(url, "utf-8");
        return JSON.parse(raw) as T;
    } catch {
        return fallback;
    }
};

const normalizationUrl = new URL("../resources/normalization.json", import.meta.url);
const mccRiskUrl = new URL("../resources/mcc_risk.json", import.meta.url);
const referencesUrl = new URL("../resources/references.bin", import.meta.url);
const labelsUrl = new URL("../resources/labels.bin", import.meta.url);

const normalization = readJson(normalizationUrl, defaultNormalization);
const mccRisk = readJson<MccRiskMap>(mccRiskUrl, {});

const dims = 14;
const k = 5;
let ready = false;
let references = new Float32Array(0);
let labels = new Uint8Array(0);

const loadReferences = () => {
    try {
        const vectorsBuffer = readFileSync(referencesUrl);
        const labelsBuffer = readFileSync(labelsUrl);

        const vectorsView = new Float32Array(
            vectorsBuffer.buffer,
            vectorsBuffer.byteOffset,
            Math.floor(vectorsBuffer.byteLength / 4)
        );
        const labelsView = new Uint8Array(
            labelsBuffer.buffer,
            labelsBuffer.byteOffset,
            labelsBuffer.byteLength
        );

        const vectorCount = Math.floor(vectorsView.length / dims);
        const usableCount = Math.min(vectorCount, labelsView.length);

        if (usableCount <= 0) {
            ready = false;
            return;
        }

        references = vectorsView.subarray(0, usableCount * dims);
        labels = labelsView.subarray(0, usableCount);
        ready = true;
    } catch {
        ready = false;
    }
};

loadReferences();

const jsonResponse = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" }
    });

const port = Number(process.env.PORT ?? "9999");

Bun.serve({
    port,
    fetch: async (request) => {
        const url = new URL(request.url);

        if (url.pathname === "/ready") {
            if (!ready) {
                return new Response(null, { status: 204 });
            }

            return new Response("OK", { status: 200 });
        }

        if (url.pathname === "/fraud-score") {
            if (request.method !== "POST") {
                return new Response(null, { status: 404 });
            }

            try {
                const payload = (await request.json()) as FraudPayload;

                if (!ready) {
                    return jsonResponse(safeResponse);
                }

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
