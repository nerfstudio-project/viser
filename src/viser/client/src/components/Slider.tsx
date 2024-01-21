import { Flex, Box, Slider, NumberInput, Text } from '@mantine/core';
import { WrapInputDefault } from "./utils";
import { GuiProps } from "../ControlPanel/GuiState";
import { SliderProps } from "../WebsocketMessages";


export default function RgbInputComponent({ disabled, value, update, ...conf }: GuiProps<SliderProps>) {
  const min = conf.min ?? 0;
  const max = conf.max ?? 100;
  return (<WrapInputDefault>
        <Flex justify="space-between">
          <Box sx={{ flexGrow: 1 }}>
            <Slider
              id={conf.id}
              size="xs"
              thumbSize={0}
              styles={(theme) => ({
                thumb: {
                  background: theme.fn.primaryColor(),
                  borderRadius: "0.1em",
                  height: "0.75em",
                  width: "0.625em",
                },
              })}
              pt="0.2em"
              showLabelOnHover={false}
              min={min}
              max={max}
              step={conf.step ?? undefined}
              precision={conf.precision ?? undefined}
              value={value}
              onChange={(value) => update({ value })}
              marks={[{ value: min }, { value: max }]}
              disabled={disabled}
            />
            <Flex
              justify="space-between"
              fz="0.6rem"
              c="dimmed"
              lh="1.2em"
              lts="-0.5px"
              mt="-0.0625em"
              mb="-0.4em"
            >
              <Text>{parseInt(min.toFixed(6))}</Text>
              <Text>{parseInt(max.toFixed(6))}</Text>
            </Flex>
          </Box>
          <NumberInput
            value={value}
            onChange={(value) => {
              // Ignore empty values.
              value !== "" && update({ value });
            }}
            size="xs"
            min={min}
            max={max}
            hideControls
            step={conf.step ?? undefined}
            precision={conf.precision ?? undefined}
            sx={{ width: "3rem" }}
            styles={{
              input: {
                padding: "0.375em",
                letterSpacing: "-0.5px",
                minHeight: "1.875em",
                height: "1.875em",
              },
            }}
            ml="xs"
          />
        </Flex>
        </WrapInputDefault>);
}