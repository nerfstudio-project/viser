import { TextInput } from "@mantine/core";
import { TextInputProps } from "../WebsocketMessages";
import { GuiProps } from "../ControlPanel/GuiState";
import { WrapInputDefault } from "./utils";

export default function TextInputComponent({ id, value, disabled, update }: GuiProps<TextInputProps>) {
  return <WrapInputDefault>
    <TextInput
      id={id}
      value={value}
      size="xs"
      onChange={(e) => update({ value: e.target.value})}
      styles={{
      input: {
          minHeight: "1.625rem",
          height: "1.625rem",
          padding: "0 0.5em",
      },
      }}
      disabled={disabled}
      />
  </WrapInputDefault>
}