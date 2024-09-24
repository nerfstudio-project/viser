import * as React from "react";
import { ColorInput } from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { rgbaToHex, hexToRgba } from "./utils";
import { ViserInputComponent } from "./common";
import { GuiRgbaMessage } from "../WebsocketMessages";

export default function RgbaComponent({
  id,
  value,
  props: { label, hint, disabled, visible },
}: GuiRgbaMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  return (
    <ViserInputComponent {...{ id, hint, label }}>
      <ColorInput
        disabled={disabled}
        size="xs"
        value={rgbaToHex(value)}
        onChange={(v) => setValue(id, hexToRgba(v))}
        format="hexa"
        // zIndex of dropdown should be >modal zIndex.
        // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
        popoverProps={{ zIndex: 1000 }}
        styles={{
          input: { height: "1.625rem", minHeight: "1.625rem" },
        }}
      />
    </ViserInputComponent>
  );
}
