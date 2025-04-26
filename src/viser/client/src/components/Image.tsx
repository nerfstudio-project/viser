import React from "react";
import { useEffect, useState } from "react";
import { GuiImageMessage } from "../WebsocketMessages";
import { Box, Text } from "@mantine/core";
import { ViewerContext } from "../ViewerContext";

function ImageComponent({ uuid, props }: GuiImageMessage) {
  if (!props.visible) return <></>;

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const viewer = React.useContext(ViewerContext)!;

  useEffect(() => {
    if (props._data === null) {
      setImageUrl(null);
    } else {
      const image_url = URL.createObjectURL(
        new Blob([props._data], { type: props.media_type }),
      );
      setImageUrl(image_url);
      return () => {
        URL.revokeObjectURL(image_url);
      };
    }
  }, [props._data, props.media_type]);

  return imageUrl === null ? null : (
    <Box px="xs">
      {props.label === null ? null : (
        <Text fz="sm" display="block">
          {props.label}
        </Text>
      )}

      <img
        src={imageUrl}
        style={{
          maxWidth: "100%",
          height: "auto",
          cursor: props._clickable === false ? "default" : "pointer",
        }}
        onClick={(e) => {
          if (props._clickable === false) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const x = (e.clientX - rect.left) / rect.width;
          const y = (e.clientY - rect.top) / rect.height;
          viewer.sendMessageRef.current({
            type: "GuiUpdateMessage",
            uuid: uuid,
            updates: { value: [x, y] },
          });
        }}
      />
    </Box>
  );
}

export default ImageComponent;
