import { Box, Progress } from "@mantine/core";
import { GuiAddProgressBarMessage } from "../WebsocketMessages";

export default function ProgressBarComponent({
  visible,
  color,
  value,
  loading,
}: GuiAddProgressBarMessage) {
  if (!visible) return <></>;
  return (
    <Box pb="xs" px="sm">
      <Progress color={color ?? undefined} value={value} striped={loading} animated={loading} />
    </Box>
  );
}
