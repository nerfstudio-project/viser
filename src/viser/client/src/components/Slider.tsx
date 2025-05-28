import React from "react";
import { GuiSliderMessage } from "../WebsocketMessages";
import {
  Slider,
  Flex,
  NumberInput,
  useMantineColorScheme,
} from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViserInputComponent } from "./common";
import { sliderDefaultMarks } from "./ComponentStyles.css";

export default function SliderComponent({
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
  },
}: GuiSliderMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  const updateValue = (value: number) => setValue(uuid, value);
  const input = (
    <Flex justify="space-between">
      <Slider
        id={uuid}
        className={marks === null ? sliderDefaultMarks : undefined}
        size="xs"
        thumbSize={0}
        radius="xs"
        style={{ flexGrow: 1 }}
        styles={{
          thumb: {
            height: "0.75rem",
            width: "0.5rem",
          },
          trackContainer: {
            zIndex: 3,
            position: "relative",
          },
          markLabel: {
            transform: "translate(-50%, 0.05rem)",
            fontSize: "0.6rem",
            textAlign: "center",
          },
          mark: {
            transform: "scale(1.95)",
          },
        }}
        pt="0.3em"
        pb="0.2em"
        showLabelOnHover={false}
        min={min}
        max={max}
        step={step ?? undefined}
        precision={precision}
        value={value}
        onChange={updateValue}
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
        disabled={disabled}
      />
      <NumberInput
        value={value}
        onChange={(newValue) => {
          // Ignore empty values.
          newValue !== "" && updateValue(Number(newValue));
        }}
        size="xs"
        min={min}
        max={max}
        hideControls
        step={step ?? undefined}
        // precision={precision}
        style={{ width: "3rem" }}
        styles={{
          input: {
            padding: "0.375em",
            letterSpacing: "-0.5px",
            minHeight: "1.875em",
            height: "1.875em",
          },
        }}
        ml="xs"
      />
    </Flex>
  );

  return (
    <ViserInputComponent {...{ uuid, hint, label }}>
      {input}
    </ViserInputComponent>
  );
}
