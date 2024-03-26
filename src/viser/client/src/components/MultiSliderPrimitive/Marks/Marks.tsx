import React from "react";
import { Box } from "@mantine/core";
import { useSliderContext } from "../Slider.context";
import { getPosition } from "../utils/get-position/get-position";

export interface MarksProps {
  marks: { value: number; label?: React.ReactNode }[] | undefined;
  min: number;
  max: number;
  value: number;
  offset: number | undefined;
  disabled: boolean | undefined;
  inverted: boolean | undefined;
}

export function Marks({
  marks,
  min,
  max,
  disabled,
  value, // eslint-disable-line
  offset, // eslint-disable-line
  inverted, // eslint-disable-line
}: MarksProps) {
  const { getStyles } = useSliderContext();

  if (!marks) {
    return null;
  }

  const items = marks.map((mark, index) => (
    <Box
      {...getStyles("markWrapper")}
      __vars={{
        "--mark-offset": `${getPosition({ value: mark.value, min, max })}%`,
      }}
      key={index}
    >
      <Box
        {...getStyles("mark")}
        mod={{
          filled: false,
          disabled,
        }}
      />
      {mark.label && <div {...getStyles("markLabel")}>{mark.label}</div>}
    </Box>
  ));

  return <div>{items}</div>;
}

Marks.displayName = "@mantine/core/SliderMarks";
