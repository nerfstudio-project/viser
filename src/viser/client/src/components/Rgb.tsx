import * as React from "react";
import { ColorInput } from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { rgbToHex, hexToRgb } from "./utils";
import { ViserInputComponent } from "./common";
import { GuiAddRgbMessage } from "../WebsocketMessages";

export default function RgbComponent({
  id,
  label,
  hint,
  value,
  disabled,
  visible,
}: GuiAddRgbMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  return (
    <ViserInputComponent {...{ id, hint, label }}>
      <ColorInput
        disabled={disabled}
        size="xs"
        value={rgbToHex(value)}
        onChange={(v) => setValue(id, hexToRgb(v))}
        format="hex"
        // zIndex of dropdown should be >modal zIndex.
        // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
        popoverProps={{zIndex: 1000}}
        styles={{
          input: { height: "1.625rem", minHeight: "1.625rem" },
          // icon: { transform: "scale(0.8)" },
        }}
      />
    </ViserInputComponent>
  );
}
