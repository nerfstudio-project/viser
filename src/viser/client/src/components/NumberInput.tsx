import { NumberInput } from "@mantine/core";
import { WrapInputDefault } from "./utils";
import { GuiProps } from "../ControlPanel/GuiState";
import { NumberInputProps } from "../WebsocketMessages";


export default function NumberInputComponent({ id, precision, min, max, step, disabled, value, update }: GuiProps<NumberInputProps>) {
  return (<WrapInputDefault>
    <NumberInput
      id={id}
      value={value}
      precision={precision ?? undefined}
      min={min ?? undefined}
      max={max ?? undefined}
      step={step ?? undefined}
      size="xs"
      onChange={(value) => {
        // Ignore empty values.
        value !== "" && update({ value });
      }}
      styles={{
        input: {
          minHeight: "1.625rem",
          height: "1.625rem",
        },
      }}
      disabled={disabled}
      stepHoldDelay={500}
      stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
    />
  </WrapInputDefault>);
}

