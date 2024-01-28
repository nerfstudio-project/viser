import { ButtonProps } from '../WebsocketMessages';
import { useComponentContext } from '../ControlPanel/GuiState';
import { computeRelativeLuminance } from "../ControlPanel/GuiState";
import { Box, Button, Image } from '@mantine/core';
import { WrapInputDefault } from './utils';
import { useMantineTheme } from '@mantine/core';


export default function ButtonComponent({ 
  onClick, 
  disabled, 
  color,
  label,
  icon_base64 
} : ButtonProps) {
  const theme = useMantineTheme();
  const id = useComponentContext<ButtonProps>();
  let inputColor =
    computeRelativeLuminance(theme.fn.primaryColor()) > 50.0
    ? theme.colors.gray[9]
    : theme.white;
    if (color !== null) {
      inputColor =
        computeRelativeLuminance(
          theme.colors[color][theme.fn.primaryShade()],
        ) > 50.0
          ? theme.colors.gray[9]
          : theme.white;
    }

  return <WrapInputDefault label={null}>
      <Button
        id={id}
        fullWidth
        color={color ?? undefined}
        onClick={onClick}
        style={{ height: "2.125em" }}
        styles={{ inner: { color: inputColor + " !important" } }}
        disabled={disabled}
        size="sm"
        leftIcon={
          icon_base64 === null ? undefined : (
            <Image
              /*^In Safari, both the icon's height and width need to be set, otherwise the icon is clipped.*/
              height="1em"
              width="1em"
              opacity={disabled ? 0.3 : 1.0}
              mr="-0.125em"
              sx={
                inputColor === theme.white
                  ? {
                      // Make the color white.
                      filter: !disabled ? "invert(1)" : undefined,
                    }
                  : // Icon will be black by default.
                    undefined
              }
              src={"data:image/svg+xml;base64," + icon_base64}
            />
          )
        }
      >
        {label}
      </Button>
  </WrapInputDefault>
}