import { Box, Tooltip } from '@mantine/core';
import { WrapInputDefault } from './utils';
import { Checkbox } from '@mantine/core';
import { CheckboxProps } from '../WebsocketMessages';
import { GuiProps } from '../ControlPanel/GuiState';
import { computeRelativeLuminance } from "../ControlPanel/GuiState";
import { useMantineTheme } from '@mantine/core';


export default function CheckboxComponent({ id, value, disabled, update, hint, ...props}: GuiProps<CheckboxProps>) {
  const theme = useMantineTheme();
  let inputColor =
    computeRelativeLuminance(theme.fn.primaryColor()) > 50.0
      ? theme.colors.gray[9]
      : theme.white;
  let input = (<>
    <Checkbox
      id={id}
      checked={value}
      size="xs"
      onChange={(value) => update({ value: value.target.checked })}
      disabled={disabled}
      styles={{
        icon: {
          color: inputColor + " !important",
        },
      }}
    />
  </>);
  if (hint !== null)
    input = // We need to add <Box /> for inputs that we can't assign refs to.
      (
        <Tooltip
          zIndex={100}
          label={hint}
          multiline
          w="15rem"
          withArrow
          openDelay={500}
          withinPortal
        >
          <Box
            sx={{
              display:
                // For checkboxes, we want to make sure that the wrapper
                // doesn't expand to the full width of the parent. This will
                // de-center the tooltip.
                  "inline-block"
            }}
          >
            {input}
          </Box>
        </Tooltip>
      );
  return <WrapInputDefault hint={null}>{input}</WrapInputDefault>;
}