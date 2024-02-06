/**
 * Read all the data from stdin as a string.
 */
export async function readStdin(): Promise<string> {
  let buf = "";
  const dec = new TextDecoder();
  for await (const chunk of Deno.stdin.readable) {
    buf += dec.decode(chunk);
  }
  return buf;
}

export function unreachable(x: never): never {
  throw "impossible case reached";
}
