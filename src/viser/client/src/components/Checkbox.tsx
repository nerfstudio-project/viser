import * as React from "react";
import { ViserInputComponent } from "./common";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { GuiAddCheckboxMessage } from "../WebsocketMessages";
import { Box, Checkbox, Tooltip, useMantineTheme } from "@mantine/core";

export default function CheckboxComponent({
  id,
  disabled,
  visible,
  hint,
  label,
  value,
}: GuiAddCheckboxMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  const theme = useMantineTheme();
  let input = (
    <Checkbox
      id={id}
      checked={value}
      size="xs"
      onChange={(value) => {
        setValue(id, value.target.checked);
      }}
      disabled={disabled}
    />
  );
  if (hint !== null && hint !== undefined) {
    // For checkboxes, we want to make sure that the wrapper
    // doesn't expand to the full width of the parent. This will
    // de-center the tooltip.
    input = (
      <Tooltip
        zIndex={100}
        label={hint}
        multiline
        w="15rem"
        withArrow
        openDelay={500}
        withinPortal
      >
        <Box style={{ display: "inline-block" }}>{input}</Box>
      </Tooltip>
    );
  }
  return <ViserInputComponent {...{ id, label }}>{input}</ViserInputComponent>;
}
