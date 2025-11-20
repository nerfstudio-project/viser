import { GuiButtonMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { Box } from "@mantine/core";

import { Button } from "@mantine/core";
import React from "react";
import { htmlIconWrapper } from "./ComponentStyles.css";
import { toMantineColor } from "./colorUtils";

export default function ButtonComponent({
  uuid,
  props: { visible, disabled, label, color, _icon_html: icon_html },
}: GuiButtonMessage) {
  const { messageSender } = React.useContext(GuiComponentContext)!;
  if (!(visible ?? true)) return null;

  // Function to send the message when mouse is pressed
  const handleMouseDown = () => {
    messageSender({
      type: "GuiUpdateMessage",
      uuid: uuid,
      updates: { value: true },
    });
  };

  // Function to send the message when mouse is released
  const handleMouseUp = () => {
    messageSender({
      type: "GuiUpdateMessage",
      uuid: uuid,
      updates: { value: false },
    });
  };

  return (
    <Box mx="xs" pb="0.5em">
      <Button
        id={uuid}
        fullWidth
        color={toMantineColor(color)}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        style={{
          height: "2em",
        }}
        disabled={disabled ?? false}
        size="sm"
        leftSection={
          icon_html === null ? undefined : (
            <div className={htmlIconWrapper} dangerouslySetInnerHTML={{ __html: icon_html }} />
          )
        }
      >
        {label}
      </Button>
    </Box>
  );
}
