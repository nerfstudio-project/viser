import React from "react";
import { GuiMultiSliderMessage } from "../WebsocketMessages";
import { Box, useMantineColorScheme } from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViserInputComponent } from "./common";
import { MultiSlider } from "./MultiSliderPrimitive";
import { sliderDefaultMarks } from "./ComponentStyles.css";

export default function MultiSliderComponent({
  uuid,
  value,
  props: {
    label,
    hint,
    visible,
    disabled,
    min,
    max,
    precision,
    step,
    _marks: marks,
    fixed_endpoints,
    min_range,
  },
}: GuiMultiSliderMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  const updateValue = (value: number[]) => setValue(uuid, value);
  const colorScheme = useMantineColorScheme().colorScheme;
  const input = (
    <Box mt="0.2em" mb="0.4em">
      <MultiSlider
        id={uuid}
        className={marks === null ? sliderDefaultMarks : undefined}
        size="xs"
        radius="xs"
        styles={(theme) => ({
          thumb: {
            height: "0.75rem",
            width: "0.5rem",
          },
          trackContainer: {
            zIndex: 3,
            position: "relative",
          },
          markLabel: {
            transform: "translate(-50%, 0.03rem)",
            fontSize: "0.6rem",
            textAlign: "center",
          },
          mark: {
            transform: "scale(1.85)",
          },
          markFilled: {
            background: disabled
              ? colorScheme === "dark"
                ? theme.colors.dark[3]
                : theme.colors.gray[4]
              : theme.primaryColor,
          },
        })}
        pt="0.2em"
        pb="0.4em"
        min={min}
        max={max}
        step={step ?? undefined}
        fixedEndpoints={fixed_endpoints}
        precision={precision}
        minRange={min_range ?? undefined}
        marks={
          marks === null
            ? [
                {
                  value: min,
                  // The regex here removes trailing zeros and the decimal
                  // point if the number is an integer.
                  label: `${min.toFixed(6).replace(/\.?0+$/, "")}`,
                },
                {
                  value: max,
                  // The regex here removes trailing zeros and the decimal
                  // point if the number is an integer.
                  label: `${max.toFixed(6).replace(/\.?0+$/, "")}`,
                },
              ]
            : marks
        }
        value={value}
        onChange={updateValue}
      />
    </Box>
  );

  return (
    <ViserInputComponent {...{ uuid, hint, label }}>
      {input}
    </ViserInputComponent>
  );
}
