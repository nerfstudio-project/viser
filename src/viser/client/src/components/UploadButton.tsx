import { GuiAddUploadButtonMessage } from "../WebsocketMessages";
import { computeRelativeLuminance } from "../ControlPanel/GuiState";
import { v4 as uuid } from "uuid";
import { Box, Image, Progress, useMantineTheme } from "@mantine/core";

import { Button } from "@mantine/core";
import React, { useContext } from "react";
import { ViewerContext, ViewerContextContents } from "../App";
import { pack } from "msgpackr";
import { IconCheck } from "@tabler/icons-react";
import { notifications } from "@mantine/notifications";

export default function UploadButtonComponent(conf: GuiAddUploadButtonMessage) {
  // Handle GUI input types.
  const viewer = useContext(ViewerContext)!;
  const fileUploadRef = React.useRef<HTMLInputElement>(null);
  const { isUploading, upload } = useFileUpload({
    viewer,
    componentId: conf.id,
  });
  const theme = useMantineTheme();

  const inputColor =
    computeRelativeLuminance(theme.primaryColor) > 50.0
      ? theme.colors.gray[9]
      : theme.white;
  const disabled = conf.disabled || isUploading;
  return (
    <Box mx="xs" mb="0.5em">
      <input
        type="file"
        style={{ display: "none" }}
        id={`file_upload_${conf.id}`}
        name="file"
        accept={conf.mime_type}
        ref={fileUploadRef}
        onChange={(e) => {
          const input = e.target as HTMLInputElement;
          if (!input.files) return;
          upload(input.files[0]);
        }}
      />
      <Button
        id={conf.id}
        fullWidth
        color={conf.color ?? undefined}
        onClick={() => {
          if (fileUploadRef.current === null) return;
          fileUploadRef.current.value = fileUploadRef.current.defaultValue;
          fileUploadRef.current.click();
        }}
        style={{ height: "2.125em" }}
        styles={{ inner: { color: inputColor + " !important" } }}
        disabled={disabled}
        size="sm"
        leftSection={
          conf.icon_base64 === null ? undefined : (
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
              src={"data:image/svg+xml;base64," + conf.icon_base64}
            />
          )
        }
      >
        {conf.label}
      </Button>
    </Box>
  );
}

function useFileUpload({
  viewer,
  componentId,
}: {
  componentId: string;
  viewer: ViewerContextContents;
}) {
  const websocketRef = viewer.websocketRef;
  const updateUploadState = viewer.useGui((state) => state.updateUploadState);
  const uploadState = viewer.useGui(
    (state) => state.uploadsInProgress[componentId],
  );
  const totalBytes = uploadState?.totalBytes;

  // Cache total bytes string
  const totalBytesString = React.useMemo(() => {
    if (totalBytes === undefined) return "";
    let displaySize = totalBytes;
    const displayUnits = ["B", "K", "M", "G", "T", "P"];
    let displayUnitIndex = 0;
    while (displaySize >= 100 && displayUnitIndex < displayUnits.length - 1) {
      displaySize /= 1024;
      displayUnitIndex += 1;
    }
    return `${displaySize.toFixed(1)}${displayUnits[displayUnitIndex]}`;
  }, [totalBytes]);

  // Update notification status
  React.useEffect(() => {
    if (uploadState === undefined) return;
    const { notificationId, filename } = uploadState;
    if (uploadState.uploadedBytes === 0) {
      // Show notification.
      notifications.show({
        id: notificationId,
        title: "Uploading " + `${filename} (${totalBytesString})`,
        message: <Progress size="sm" value={0} />,
        autoClose: false,
        withCloseButton: false,
        loading: true,
      });
    } else {
      // Update progress.
      const progressValue = uploadState.uploadedBytes / uploadState.totalBytes;
      const isDone = progressValue === 1.0;
      notifications.update({
        id: notificationId,
        title: "Uploading " + `${filename} (${totalBytesString})`,
        message: !isDone ? (
          <Progress
            size="sm"
            transitionDuration={10}
            value={100 * progressValue}
          />
        ) : (
          "File uploaded successfully."
        ),
        autoClose: isDone,
        withCloseButton: isDone,
        loading: !isDone,
        icon: isDone ? <IconCheck /> : undefined,
      });
    }
  }, [uploadState, totalBytesString]);

  const isUploading =
    uploadState !== undefined &&
    uploadState.uploadedBytes < uploadState.totalBytes;

  async function upload(file: File) {
    const chunkSize = 512 * 1024; // bytes
    const numChunks = Math.ceil(file.size / chunkSize);
    const transferUuid = uuid();
    const notificationId = "upload-" + transferUuid;

    const send = (message: Parameters<typeof pack>[0]) =>
      websocketRef.current?.send(pack(message));

    // Begin upload by setting initial state
    updateUploadState({
      componentId: componentId,
      uploadedBytes: 0,
      totalBytes: file.size,
      filename: file.name,
      notificationId,
    });

    send({
      type: "FileTransferStart",
      source_component_id: componentId,
      transfer_uuid: transferUuid,
      filename: file.name,
      mime_type: file.type,
      size_bytes: file.size,
      part_count: numChunks,
    });

    for (let i = 0; i < numChunks; i++) {
      const start = i * chunkSize;
      const end = (i + 1) * chunkSize;
      const chunk = file.slice(start, end);
      const buffer = await chunk.arrayBuffer();

      send({
        type: "FileTransferPart",
        source_component_id: componentId,
        transfer_uuid: transferUuid,
        part: i,
        content: new Uint8Array(buffer),
      });
    }
  }

  return {
    isUploading,
    upload,
  };
}
