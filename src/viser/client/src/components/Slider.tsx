import React from "react";
import { GuiAddSliderMessage } from "../WebsocketMessages";
import { Slider, Flex, NumberInput } from "@mantine/core";
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
  const input = (
    <Flex justify="space-between">
      <Slider
        id={id}
        size="xs"
        thumbSize={0}
        style={{ flexGrow: 1 }}
        styles={(theme) => ({
          thumb: {
            background: theme.fn.primaryColor(),
            borderRadius: "0.1rem",
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
          marksContainer: {
            left: "0.2rem",
            right: "0.2rem",
          },
          markWrapper: {
            position: "absolute",
            top: `0.03rem`,
            ...(marks === null
              ? /*  Shift the mark labels so they don't spill too far out the left/right when we only have min and max marks. */
                {
                  ":first-child": {
                    "div:nth-child(2)": {
                      transform: "translate(-0.2rem, 0.03rem)",
                    },
                  },
                  ":last-child": {
                    "div:nth-child(2)": {
                      transform: "translate(-90%, 0.03rem)",
                    },
                  },
                }
              : {}),
          },
          mark: {
            border: "0px solid transparent",
            background:
              theme.colorScheme === "dark"
                ? theme.colors.dark[4]
                : theme.colors.gray[2],
            width: "0.42rem",
            height: "0.42rem",
            transform: `translateX(-50%)`,
          },
          markFilled: {
            background: disabled
              ? theme.colorScheme === "dark"
                ? theme.colors.dark[3]
                : theme.colors.gray[4]
              : theme.fn.primaryColor(),
          },
        })}
        pt="0.2em"
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
          newValue !== "" && updateValue(newValue);
        }}
        size="xs"
        min={min}
        max={max}
        hideControls
        step={step ?? undefined}
        precision={precision}
        sx={{ width: "3rem" }}
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

  const containerProps = {};

  return (
    <ViserInputComponent {...{ id, hint, label }}>{input}</ViserInputComponent>
  );
}
