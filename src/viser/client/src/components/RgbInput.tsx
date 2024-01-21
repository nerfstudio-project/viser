import { ColorInput } from "@mantine/core";
import { hexToRgb, rgbToHex, WrapInputDefault } from "./utils";
import { GuiProps } from "../ControlPanel/GuiState";
import { RgbInputProps } from "../WebsocketMessages";


export default function RgbInputComponent({ disabled, value, update }: GuiProps<RgbInputProps>) {
  return (<WrapInputDefault>
    <ColorInput
      disabled={disabled}
      size="xs"
      value={rgbToHex(value)}
      onChange={(v) => update({ value: hexToRgb(v) })}
      format="hex"
      // zIndex of dropdown should be >modal zIndex.
      // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
      dropdownZIndex={1000}
      withinPortal={true}
      styles={{
        input: { height: "1.625rem", minHeight: "1.625rem" },
        icon: { transform: "scale(0.8)" },
      }}
    />
  </WrapInputDefault>);
}