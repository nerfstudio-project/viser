import { Select } from "@mantine/core";
import { WrapInputDefault } from "./utils";
import { GuiProps } from "../ControlPanel/GuiState";
import { DropdownProps } from "../WebsocketMessages";


export default function DropdownComponent({ value, update, ...conf }: GuiProps<DropdownProps>) {
  return (<WrapInputDefault>
    <Select
      id={conf.id}
      radius="xs"
      value={value}
      data={conf.options}
      onChange={(value) => update({ value })}
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
      withinPortal={true}
    />
  </WrapInputDefault>);
}

