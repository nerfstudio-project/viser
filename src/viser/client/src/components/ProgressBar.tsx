import { Box, Progress } from "@mantine/core";
import { GuiProgressBarMessage } from "../WebsocketMessages";

export default function ProgressBarComponent({
  value,
  props: { visible, color, animated },
}: GuiProgressBarMessage) {
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
