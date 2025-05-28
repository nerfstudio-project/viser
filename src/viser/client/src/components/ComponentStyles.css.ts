import { globalStyle, style } from "@vanilla-extract/css";

export const htmlIconWrapper = style({
  height: "1em",
  width: "1em",
  position: "relative",
});

globalStyle(`${htmlIconWrapper} svg`, {
  height: "auto",
  width: "1em",
  position: "absolute",
  top: "50%",
  transform: "translateY(-50%)",
});

// Class for sliders with default min/max marks. We use this for aestheticn
// its; global styles are used to shift the min/max mark labels to stay closer
// within the bounds of the slider.
export const sliderDefaultMarks = style({});

globalStyle(
  `${sliderDefaultMarks} .mantine-Slider-markWrapper:first-of-type div:nth-of-type(2)`,
  {
    transform: "translate(-0.1rem, 0.03rem) !important",
  },
);

globalStyle(
  `${sliderDefaultMarks} .mantine-Slider-markWrapper:last-of-type div:nth-of-type(2)`,
  {
    transform: "translate(-85%, 0.03rem) !important",
  },
);

// Style for filled slider marks - use primary color to match the active segment.
globalStyle(".mantine-Slider-mark[data-filled]:not([data-disabled])", {
  background: "var(--mantine-primary-color-filled)",
  borderColor: "var(--mantine-primary-color-filled)",
});

// Style for filled slider marks when disabled - use separate rules for light/dark.
globalStyle(".mantine-Slider-mark[data-filled][data-disabled]", {
  background: "var(--mantine-color-gray-5)",
  borderColor: "var(--mantine-color-gray-5)",
});

// Dark mode styles for filled marks when disabled.
globalStyle(
  '[data-mantine-color-scheme="dark"] .mantine-Slider-mark[data-filled][data-disabled]',
  {
    background: "var(--mantine-color-dark-3)",
    borderColor: "var(--mantine-color-dark-3)",
  },
);
