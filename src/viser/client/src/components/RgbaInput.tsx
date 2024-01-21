import { ColorInput } from "@mantine/core";
import { hexToRgba, rgbaToHex, WrapInputDefault } from "./utils";
import { GuiProps } from "../ControlPanel/GuiState";
import { RgbaInputProps } from "../WebsocketMessages";


export default function RgbaInputComponent({ disabled, value, update }: GuiProps<RgbaInputProps>) {
  return (<WrapInputDefault>
    <ColorInput
      disabled={disabled}
      size="xs"
      value={rgbaToHex(value)}
      onChange={(v) => update({ value: hexToRgba(v)})}
      format="hexa"
      // zIndex of dropdown should be >modal zIndex.
      // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
      dropdownZIndex={1000}
      withinPortal={true}
      styles={{ input: { height: "1.625rem", minHeight: "1.625rem" } }}
    />
  </WrapInputDefault>);
}