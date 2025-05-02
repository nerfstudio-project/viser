/**
 * Convert an RGB tuple to a hex color string or handle null value
 * @param color RGB tuple, string color name, or null
 * @returns A hex color string (e.g. "#ff0000"), the original string color name, or undefined for null
 */
export function toMantineColor(
  color: [number, number, number] | string | null,
): string | undefined {
  // Handle null case.
  if (color === null) {
    return undefined;
  }

  // If color is an RGB tuple, convert to hex.
  if (Array.isArray(color)) {
    const [r, g, b] = color;
    return `#${r.toString(16).padStart(2, "0")}${g
      .toString(16)
      .padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
  }

  // If it's already a string, return as is.
  return color;
}
