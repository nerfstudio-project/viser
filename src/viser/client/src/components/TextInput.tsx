import * as React from "react";
import { TextInput } from "@mantine/core";
import { ViserInputComponent } from "./common";
import { GuiAddTextMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";

export default function TextInputComponent(props: GuiAddTextMessage) {
  const { id, hint, label, value, disabled, visible } = props;
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  return <ViserInputComponent {...{ id, hint, label }}>
    <TextInput
      value={value}
      size="xs"
      onChange={(value) => {
        setValue(id, value.target.value);
      }}
      styles={{
        input: {
          minHeight: "1.625rem",
          height: "1.625rem",
          padding: "0 0.5em",
        },
      }}
      disabled={disabled}
    />
  </ViserInputComponent>;
}