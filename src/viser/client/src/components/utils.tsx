// Color conversion helpers.

export function rgbToHex([r, g, b]: [number, number, number]): string {
  const hexR = r.toString(16).padStart(2, "0");
  const hexG = g.toString(16).padStart(2, "0");
  const hexB = b.toString(16).padStart(2, "0");
  return `#${hexR}${hexG}${hexB}`;
}

export function hexToRgb(hexColor: string): [number, number, number] {
  const hex = hexColor.slice(1); // Remove the # in #ffffff.
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return [r, g, b];
}
export function rgbaToHex([r, g, b, a]: [
  number,
  number,
  number,
  number,
]): string {
  const hexR = r.toString(16).padStart(2, "0");
  const hexG = g.toString(16).padStart(2, "0");
  const hexB = b.toString(16).padStart(2, "0");
  const hexA = a.toString(16).padStart(2, "0");
  return `#${hexR}${hexG}${hexB}${hexA}`;
}

export function hexToRgba(hexColor: string): [number, number, number, number] {
  const hex = hexColor.slice(1); // Remove the # in #ffffff.
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  const a = parseInt(hex.substring(6, 8), 16);
  return [r, g, b, a];
}
