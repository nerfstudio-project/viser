import React from "react";
import { GuiAddSliderMessage } from "../WebsocketMessages";
import { Slider, Box, Flex, Text, NumberInput } from "@mantine/core";
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
  const { min, max, precision, step } = otherProps;
  let input = (
    <React.Fragment>
      <Box sx={{ flexGrow: 1 }}>
        <Slider
          id={id}
          size="xs"
          thumbSize={0}
          styles={(theme) => ({
            thumb: {
              background: theme.fn.primaryColor(),
              borderRadius: "0.1em",
              height: "0.75em",
              width: "0.625em",
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
          marks={[{ value: min }, { value: max }]}
          disabled={disabled}
        />
        <Flex
          justify="space-between"
          fz="0.6rem"
          c="dimmed"
          lh="1.2em"
          lts="-0.5px"
          mt="-0.0625em"
          mb="-0.4em"
        >
          <Text>{parseInt(min.toFixed(6))}</Text>
          <Text>{parseInt(max.toFixed(6))}</Text>
        </Flex>
      </Box>
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
    </React.Fragment>
  );

  const containerProps = {};
  // if (marks?.some(x => x.label))
  //   containerProps = { ...containerProps, "mb": "md" };

  input = (
    <Flex justify="space-between" {...containerProps}>
      {input}
    </Flex>
  );
  return (
    <ViserInputComponent {...{ id, hint, label }}>{input}</ViserInputComponent>
  );
}
