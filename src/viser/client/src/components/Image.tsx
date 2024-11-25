import { useEffect, useState } from "react";
import { GuiImageMessage } from "../WebsocketMessages";
import { Box, Text } from "@mantine/core";

function ImageComponent({ props }: GuiImageMessage) {
  if (!props.visible) return <></>;

  const [imageUrl, setImageUrl] = useState<string | null>(null);

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
        }}
      />
    </Box>
  );
}

export default ImageComponent;
