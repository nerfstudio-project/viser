import * as React from "react";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViserInputComponent } from "./common";
import { GuiAddDropdownMessage } from "../WebsocketMessages";
import { Select } from "@mantine/core";

export default function DropdownComponent({
  id,
  hint,
  label,
  value,
  disabled,
  visible,
  options,
}: GuiAddDropdownMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  return (
    <ViserInputComponent {...{ id, hint, label }}>
      <Select
        id={id}
        radius="xs"
        value={value}
        data={options}
        onChange={(value) => setValue(id, value)}
        disabled={disabled}
        searchable
        maxDropdownHeight={400}
        size="xs"
        styles={{
          input: {
            padding: "0.5em",
            letterSpacing: "-0.5px",
            minHeight: "1.625rem",
            height: "1.625rem",
          },
        }}
        // zIndex of dropdown should be >modal zIndex.
        // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
        zIndex={1000}
        withinPortal
      />
    </ViserInputComponent>
  );
}
