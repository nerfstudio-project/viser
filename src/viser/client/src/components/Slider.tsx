import React from "react";
import { GuiAddSliderMessage } from "../WebsocketMessages";
import {
  Slider,
  Flex,
  NumberInput,
  useMantineColorScheme,
} from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViserInputComponent } from "./common";

export default function SliderComponent({
  id,
  label,
  hint,
  visible,
  disabled,
  value,
  ...otherProps
}: GuiAddSliderMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  const updateValue = (value: number) => setValue(id, value);
  const { min, max, precision, step, marks } = otherProps;
  const colorScheme = useMantineColorScheme().colorScheme;
  const input = (
    <Flex justify="space-between">
      <Slider
        id={id}
        size="xs"
        thumbSize={0}
        radius="xs"
        style={{ flexGrow: 1 }}
        styles={(theme) => ({
          thumb: {
            height: "0.75rem",
            width: "0.625rem",
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
            transform: "scale(1.95)",
          },
          markFilled: {
            background: disabled
              ? colorScheme === "dark"
                ? theme.colors.dark[3]
                : theme.colors.gray[4]
              : theme.primaryColor,
          },
        })}
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
                  label: `${parseInt(min.toFixed(6))}`,
                },
                {
                  value: max,
                  label: `${parseInt(max.toFixed(6))}`,
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
    <ViserInputComponent {...{ id, hint, label }}>{input}</ViserInputComponent>
  );
}
