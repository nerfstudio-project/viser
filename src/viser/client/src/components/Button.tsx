import { GuiButtonMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { Box } from "@mantine/core";

import { Button } from "@mantine/core";
import React from "react";
import { htmlIconWrapper } from "./ComponentStyles.css";

export default function ButtonComponent({
  id,
  props: { visible, disabled, label, ...otherProps },
}: GuiButtonMessage) {
  const { messageSender } = React.useContext(GuiComponentContext)!;
  const { color, icon_html } = otherProps;
  if (!(visible ?? true)) return <></>;

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
        style={{
          height: "2.125em",
        }}
        disabled={disabled ?? false}
        size="sm"
        leftSection={
          icon_html === null ? undefined : (
            <div
              className={htmlIconWrapper}
              dangerouslySetInnerHTML={{ __html: icon_html }}
            />
          )
        }
      >
        {label}
      </Button>
    </Box>
  );
}
