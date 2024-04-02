import * as React from "react";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { GuiAddNumberMessage } from "../WebsocketMessages";
import { ViserInputComponent } from "./common";
import { NumberInput } from "@mantine/core";

export default function NumberInputComponent({
  visible,
  id,
  label,
  hint,
  value,
  disabled,
  ...otherProps
}: GuiAddNumberMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  const { precision, min, max, step } = otherProps;
  if (!visible) return <></>;
  return (
    <ViserInputComponent {...{ id, hint, label }}>
      <NumberInput
        id={id}
        value={value}
        // This was renamed in Mantine v7.
        decimalScale={precision}
        min={min ?? undefined}
        max={max ?? undefined}
        step={step}
        size="xs"
        onChange={(newValue) => {
          // Ignore empty values.
          newValue !== "" && setValue(id, newValue);
        }}
        styles={{
          input: {
            minHeight: "1.625rem",
            height: "1.625rem",
          },
          controls: {
            height: "1.625em",
            width: "0.825em",
          },
        }}
        disabled={disabled}
        stepHoldDelay={500}
        stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
      />
    </ViserInputComponent>
  );
}
