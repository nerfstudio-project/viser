import * as React from "react";
import { TextInput, Textarea } from "@mantine/core";
import { ViserInputComponent } from "./common";
import { GuiTextMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";

export default function TextInputComponent({
  uuid,
  value,
  props: { hint, label, disabled, visible, multiline },
}: GuiTextMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  return (
    <ViserInputComponent {...{ uuid, hint, label }}>
      {multiline ? (
        <Textarea
          value={value}
          size="xs"
          onChange={(value) => {
            setValue(uuid, value.target.value);
          }}
          styles={{
            input: {
              minHeight: "1.625rem",
              padding: "0 0.5em",
            },
          }}
          disabled={disabled}
          autosize
        />
      ) : (
        <TextInput
          value={value}
          size="xs"
          onChange={(value) => {
            setValue(uuid, value.target.value);
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
      )}
    </ViserInputComponent>
  );
}
