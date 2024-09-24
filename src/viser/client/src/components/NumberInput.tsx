import * as React from "react";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { GuiNumberMessage } from "../WebsocketMessages";
import { ViserInputComponent } from "./common";
import { NumberInput } from "@mantine/core";

export default function NumberInputComponent({
  id,
  value,
  props: { visible, label, hint, disabled, precision, min, max, step },
}: GuiNumberMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
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
