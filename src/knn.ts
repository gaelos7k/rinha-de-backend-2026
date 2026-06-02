const swap = (
    distances: Float64Array,
    labels: Uint8Array,
    left: number,
    right: number
) => {
    const distLeft = distances[left];
    const distRight = distances[right];
    const labelLeft = labels[left];
    const labelRight = labels[right];

    if (
        distLeft === undefined ||
        distRight === undefined ||
        labelLeft === undefined ||
        labelRight === undefined
    ) {
        return;
    }

    distances[left] = distRight;
    distances[right] = distLeft;
    labels[left] = labelRight;
    labels[right] = labelLeft;
};

class MaxHeap {
    private readonly distances: Float64Array;
    private readonly labels: Uint8Array;
    private size = 0;

    constructor(private readonly capacity: number) {
        this.distances = new Float64Array(capacity);
        this.labels = new Uint8Array(capacity);
    }

    getSize(): number {
        return this.size;
    }

    peek(): number {
        if (this.size === 0) {
            return Infinity;
        }

        const value = this.distances[0];
        return value === undefined ? Infinity : value;
    }

    push(distance: number, label: number): void {
        const index = this.size;
        if (index >= this.capacity) {
            return;
        }

        this.distances[index] = distance;
        this.labels[index] = label;
        this.size += 1;
        this.bubbleUp(index);
    }

    replaceRoot(distance: number, label: number): void {
        if (this.size === 0) {
            this.push(distance, label);
            return;
        }

        this.distances[0] = distance;
        this.labels[0] = label;
        this.bubbleDown(0);
    }

    countFraud(): number {
        let count = 0;
        for (let i = 0; i < this.size; i += 1) {
            const value = this.labels[i];
            if (value !== undefined) {
                count += value;
            }
        }
        return count;
    }

    private bubbleUp(index: number): void {
        let current = index;
        while (current > 0) {
            const parent = (current - 1) >> 1;
            const parentValue = this.distances[parent];
            const currentValue = this.distances[current];
            if (parentValue === undefined || currentValue === undefined) {
                return;
            }

            if (parentValue >= currentValue) {
                break;
            }
            swap(this.distances, this.labels, parent, current);
            current = parent;
        }
    }

    private bubbleDown(index: number): void {
        let current = index;
        while (true) {
            const left = current * 2 + 1;
            if (left >= this.size) {
                break;
            }

            const right = left + 1;
            let largest = left;
            const leftValue = this.distances[left];
            if (leftValue === undefined) {
                return;
            }

            if (right < this.size) {
                const rightValue = this.distances[right];
                if (rightValue !== undefined && rightValue > leftValue) {
                    largest = right;
                }
            }

            const currentValue = this.distances[current];
            const largestValue = this.distances[largest];
            if (currentValue === undefined || largestValue === undefined) {
                return;
            }

            if (currentValue >= largestValue) {
                break;
            }

            swap(this.distances, this.labels, current, largest);
            current = largest;
        }
    }
}

export const knnFraudScore = (
    query: Float32Array,
    references: Uint8Array,
    labels: Uint8Array,
    k = 5,
    dims = 14
): number => {
    if (k <= 0 || query.length < dims) {
        return 0;
    }

    const totalVectors = Math.min(
        Math.floor(references.length / dims),
        labels.length
    );

    if (totalVectors === 0) {
        return 0;
    }

    const heap = new MaxHeap(k);

    for (let i = 0; i < totalVectors; i += 1) {
        const offset = i * dims;
        let distance = 0;

        if (heap.getSize() === k) {
            const limit = heap.peek();
            for (let d = 0; d < dims; d += 1) {
                const queryValue = query[d];
                const referenceByte = references[offset + d];
                if (queryValue === undefined || referenceByte === undefined) {
                    return 0;
                }

                let referenceValue = referenceByte / 255;
                if ((d === 5 || d === 6) && referenceByte === 255) {
                    referenceValue = -1;
                }

                const diff = queryValue - referenceValue;
                distance += diff * diff;
                if (distance >= limit) {
                    break;
                }
            }

            if (distance >= limit) {
                continue;
            }

            const label = labels[i];
            if (label === undefined) {
                return 0;
            }

            heap.replaceRoot(distance, label);
            continue;
        }

        for (let d = 0; d < dims; d += 1) {
            const queryValue = query[d];
            const referenceByte = references[offset + d];
            if (queryValue === undefined || referenceByte === undefined) {
                return 0;
            }

            let referenceValue = referenceByte / 255;
            if ((d === 5 || d === 6) && referenceByte === 255) {
                referenceValue = -1;
            }

            const diff = queryValue - referenceValue;
            distance += diff * diff;
        }

        const label = labels[i];
        if (label === undefined) {
            return 0;
        }

        heap.push(distance, label);
    }

    return heap.countFraud() / k;
};
