import { GuiButtonMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { Box } from "@mantine/core";

import { Button } from "@mantine/core";
import React, { useRef, useCallback } from "react";
import { htmlIconWrapper } from "./ComponentStyles.css";
import { toMantineColor } from "./colorUtils";

export default function ButtonComponent({
  uuid,
  props: {
    visible,
    disabled,
    label,
    color,
    _icon_html: icon_html,
    _hold_callback_freqs: holdCallbackFreqs,
  },
}: GuiButtonMessage) {
  const { messageSender } = React.useContext(GuiComponentContext)!;

  // Track interval timers for hold callbacks.
  const holdIntervalsRef = useRef<Map<number, ReturnType<typeof setInterval>>>(
    new Map(),
  );

  // Send a hold message for a specific frequency.
  const sendHoldMessage = useCallback(
    (frequency: number) => {
      messageSender({
        type: "GuiButtonHoldMessage",
        uuid: uuid,
        frequency: frequency,
      });
    },
    [messageSender, uuid],
  );

  // Start hold timers when mouse is pressed.
  const handleMouseDown = useCallback(() => {
    // Start interval timers for each registered frequency.
    for (const freq of holdCallbackFreqs) {
      // Send immediately on press.
      sendHoldMessage(freq);

      // Then start interval timer.
      const intervalMs = 1000 / freq;
      const intervalId = setInterval(() => {
        sendHoldMessage(freq);
      }, intervalMs);
      holdIntervalsRef.current.set(freq, intervalId);
    }
  }, [holdCallbackFreqs, sendHoldMessage]);

  // Stop all hold timers.
  const stopHoldTimers = useCallback(() => {
    for (const intervalId of holdIntervalsRef.current.values()) {
      clearInterval(intervalId);
    }
    holdIntervalsRef.current.clear();
  }, []);

  if (!(visible ?? true)) return null;

  return (
    <Box mx="xs" pb="0.5em">
      <Button
        id={uuid}
        fullWidth
        color={toMantineColor(color)}
        onClick={() =>
          messageSender({
            type: "GuiUpdateMessage",
            uuid: uuid,
            updates: { value: true },
          })
        }
        onMouseDown={holdCallbackFreqs.length > 0 ? handleMouseDown : undefined}
        onMouseUp={holdCallbackFreqs.length > 0 ? stopHoldTimers : undefined}
        onMouseLeave={holdCallbackFreqs.length > 0 ? stopHoldTimers : undefined}
        style={{
          height: "2em",
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
