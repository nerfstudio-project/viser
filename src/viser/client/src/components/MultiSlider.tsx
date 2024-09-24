import React from "react";
import { GuiMultiSliderMessage } from "../WebsocketMessages";
import { Box, useMantineColorScheme } from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViserInputComponent } from "./common";
import { MultiSlider } from "./MultiSliderPrimitive";
import { sliderDefaultMarks } from "./ComponentStyles.css";

export default function MultiSliderComponent({
  id,
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
    marks,
    fixed_endpoints,
    min_range,
  },
}: GuiMultiSliderMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  const updateValue = (value: number[]) => setValue(id, value);
  const colorScheme = useMantineColorScheme().colorScheme;
  const input = (
    <Box mt="0.2em" mb="0.4em">
      <MultiSlider
        id={id}
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
                  label: `${parseInt(min.toFixed(6))}`,
                },
                {
                  value: max,
                  label: `${parseInt(max.toFixed(6))}`,
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
    <ViserInputComponent {...{ id, hint, label }}>{input}</ViserInputComponent>
  );
}
