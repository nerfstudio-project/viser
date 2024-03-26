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
