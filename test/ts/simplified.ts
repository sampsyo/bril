erronious(1n);

// This snippet checks that the ts compiler doesn't try to create an unreachable else branch
function erronious(x: bigint): bigint {
    if (true) {
        return x;
    }
}
