import { GuiAddButtonMessage } from "../WebsocketMessages";
import { computeRelativeLuminance } from "../ControlPanel/GuiState";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { Box, Image, useMantineTheme } from "@mantine/core";

import { Button } from "@mantine/core";
import React from "react";

export default function ButtonComponent({
  id,
  visible,
  disabled,
  label,
  ...otherProps
}: GuiAddButtonMessage) {
  const { messageSender } = React.useContext(GuiComponentContext)!;
  const theme = useMantineTheme();
  const { color, icon_base64 } = otherProps;
  if (!(visible ?? true)) return <></>;

  const inputColor =
    computeRelativeLuminance(theme.primaryColor) > 50.0
      ? theme.colors.gray[9]
      : theme.white;
  return (
    <Box mx="xs" mb="0.5em">
      <Button
        id={id}
        fullWidth
        color={color ?? undefined}
        onClick={() =>
          messageSender({
            type: "GuiUpdateMessage",
            id: id,
            updates: { value: true },
          })
        }
        style={{ height: "2.125em" }}
        styles={{ inner: { color: inputColor + " !important" } }}
        disabled={disabled ?? false}
        size="sm"
        leftSection={
          icon_base64 === null ? undefined : (
            <Image
              style={{
                height: "1em",
                width: "1em",
                opacity: disabled ? 0.3 : 1.0,
                // Make the color white.
                filter:
                  inputColor === theme.white && !disabled
                    ? "invert(1)"
                    : undefined,
              }}
              mr="-0.125em"
              src={"data:image/svg+xml;base64," + icon_base64}
            />
          )
        }
      >
        {label}
      </Button>
    </Box>
  );
}
