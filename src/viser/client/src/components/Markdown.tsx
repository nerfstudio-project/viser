import { Box, Text } from "@mantine/core";
import Markdown from "../Markdown";
import { ErrorBoundary } from "react-error-boundary";
import { GuiAddMarkdownMessage } from "../WebsocketMessages";

    
export default function MarkdownComponent({ visible, markdown }: GuiAddMarkdownMessage) {
  if (!visible) return <></>;
  return (
    <Box pb="xs" px="sm" style={{ maxWidth: "95%" }}>
      <ErrorBoundary
        fallback={<Text align="center">Markdown Failed to Render</Text>}
      >
        <Markdown>{markdown}</Markdown>
      </ErrorBoundary>
    </Box>
  );
}