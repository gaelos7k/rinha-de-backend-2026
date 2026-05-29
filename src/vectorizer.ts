import type { FraudPayload, MccRiskMap, NormalizationConfig } from "./types";

const clamp = (value: number): number => {
    if (!Number.isFinite(value)) {
        return value > 0 ? 1 : 0;
    }

    if (value < 0) {
        return 0;
    }

    if (value > 1) {
        return 1;
    }

    return value;
};

export const vectorize = (
    payload: FraudPayload,
    normalization: NormalizationConfig,
    mccRisk: MccRiskMap
): Float32Array => {
    const { transaction, customer, merchant, terminal, last_transaction } = payload;
    const requestedAt = new Date(transaction.requested_at);
    const requestedTime = requestedAt.getTime();
    const hourOfDay = Number.isFinite(requestedTime) ? requestedAt.getUTCHours() : 0;
    const dayOfWeek = Number.isFinite(requestedTime) ? requestedAt.getUTCDay() : 0;
    const dayIndex = (dayOfWeek + 6) % 7; // Monday = 0, Sunday = 6

    let minutesSinceLast = -1;
    let kmFromLast = -1;

    if (last_transaction) {
        const lastAt = new Date(last_transaction.timestamp);
        const lastTime = lastAt.getTime();

        if (Number.isFinite(lastTime) && Number.isFinite(requestedTime)) {
            const minutesDiff = (requestedTime - lastTime) / 60000;
            minutesSinceLast = clamp(minutesDiff / normalization.max_minutes);
            kmFromLast = clamp(last_transaction.km_from_current / normalization.max_km);
        }
    }

    const amountVsAvg =
        customer.avg_amount === 0
            ? Infinity
            : transaction.amount / customer.avg_amount / normalization.amount_vs_avg_ratio;

    const vector = new Float32Array(14);
    vector[0] = clamp(transaction.amount / normalization.max_amount);
    vector[1] = clamp(transaction.installments / normalization.max_installments);
    vector[2] = clamp(amountVsAvg);
    vector[3] = hourOfDay / 23;
    vector[4] = dayIndex / 6;
    vector[5] = minutesSinceLast;
    vector[6] = kmFromLast;
    vector[7] = clamp(terminal.km_from_home / normalization.max_km);
    vector[8] = clamp(customer.tx_count_24h / normalization.max_tx_count_24h);
    vector[9] = terminal.is_online ? 1 : 0;
    vector[10] = terminal.card_present ? 1 : 0;
    vector[11] = customer.known_merchants.includes(merchant.id) ? 0 : 1;
    vector[12] = mccRisk[merchant.mcc] ?? 0.5;
    vector[13] = clamp(merchant.avg_amount / normalization.max_merchant_avg_amount);

    return vector;
};
