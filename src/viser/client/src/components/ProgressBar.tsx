import { Box, Progress } from "@mantine/core";
import { GuiAddProgressBarMessage } from "../WebsocketMessages";

export default function ProgressBarComponent({
  visible,
  color,
  value,
  animated,
}: GuiAddProgressBarMessage) {
  if (!visible) return <></>;
  return (
    <Box pb="xs" px="xs">
      <Progress
        radius="xs"
        color={color ?? undefined}
        value={value}
        animated={animated}
        transitionDuration={0}
      />
    </Box>
  );
}
