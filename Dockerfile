FROM oven/bun:1 AS builder

WORKDIR /app
COPY package.json ./
RUN bun install

COPY . .

RUN bun run scripts/preprocess.ts

FROM oven/bun:1-slim AS runner

WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/src ./src
COPY --from=builder /app/resources/references.bin ./resources/references.bin
COPY --from=builder /app/resources/labels.bin ./resources/labels.bin
COPY --from=builder /app/resources/mcc_risk.json ./resources/mcc_risk.json
COPY --from=builder /app/resources/normalization.json ./resources/normalization.json

EXPOSE 9999
CMD ["bun", "run", "src/server.ts"]
