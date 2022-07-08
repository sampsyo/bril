/**
 * Read all the data from stdin as a string.
 */
export async function readStdin(): Promise<string> {
  const buf = await Deno.readAll(Deno.stdin);
  return (new TextDecoder()).decode(buf);
}

export function unreachable(x: never): never {
  throw "impossible case reached";
}
