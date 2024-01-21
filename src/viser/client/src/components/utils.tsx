import React from 'react';
import { Box, Tooltip, Flex, NumberInput, Text } from '@mantine/core';
import { GuiGenerateContext } from '../ControlPanel/GuiState';


export function WrapInputDefault({ children, label, hint }: { children: React.ReactNode, label?: string | null, hint?: string | null }) {
  const { id, folderDepth, _hint, _label } = React.useContext(GuiGenerateContext)!;
  label = label === undefined ? _label : label;
  hint = hint === undefined ? _hint : hint;
  if (hint !== null && hint !== undefined)
    children = // We need to add <Box /> for inputs that we can't assign refs to.
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
          <Box sx={{ display: "block" }}>
            {children}
          </Box>
        </Tooltip>
      );

  if (label !== null && label !== undefined) {
    children = (
      <LabeledInput
        id={id}
        label={label}
        input={children}
        folderDepth={folderDepth}
      />
    );
  }

  return (
    <Box pb="0.5em" px="xs">
      {children}
    </Box>
  );
}


// Color conversion helpers.
export function rgbToHex([r, g, b]: [number, number, number]): string {
  const hexR = r.toString(16).padStart(2, "0");
  const hexG = g.toString(16).padStart(2, "0");
  const hexB = b.toString(16).padStart(2, "0");
  return `#${hexR}${hexG}${hexB}`;
}

export function hexToRgb(hexColor: string): [number, number, number] {
  const hex = hexColor.slice(1); // Remove the # in #ffffff.
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return [r, g, b];
}
export function rgbaToHex([r, g, b, a]: [number, number, number, number]): string {
  const hexR = r.toString(16).padStart(2, "0");
  const hexG = g.toString(16).padStart(2, "0");
  const hexB = b.toString(16).padStart(2, "0");
  const hexA = a.toString(16).padStart(2, "0");
  return `#${hexR}${hexG}${hexB}${hexA}`;
}

export function hexToRgba(hexColor: string): [number, number, number, number] {
  const hex = hexColor.slice(1); // Remove the # in #ffffff.
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  const a = parseInt(hex.substring(6, 8), 16);
  return [r, g, b, a];
}


export function VectorInput(
  props:
    | {
        id: string;
        n: 2;
        value: [number, number];
        min: [number, number] | null;
        max: [number, number] | null;
        step: number;
        precision?: number;
        onChange: (value: number[]) => void;
        disabled: boolean;
      }
    | {
        id: string;
        n: 3;
        value: [number, number, number];
        min: [number, number, number] | null;
        max: [number, number, number] | null;
        step: number;
        precision?: number;
        onChange: (value: number[]) => void;
        disabled: boolean;
      },
) {
  return (
    <Flex justify="space-between" style={{ columnGap: "0.5em" }}>
      {[...Array(props.n).keys()].map((i) => (
        <NumberInput
          id={i === 0 ? props.id : undefined}
          key={i}
          value={props.value[i]}
          onChange={(v) => {
            const updated = [...props.value];
            updated[i] = v === "" ? 0.0 : v;
            props.onChange(updated);
          }}
          size="xs"
          styles={{
            root: { flexGrow: 1, width: 0 },
            input: {
              paddingLeft: "0.5em",
              paddingRight: "1.75em",
              textAlign: "right",
              minHeight: "1.875em",
              height: "1.875em",
            },
            rightSection: { width: "1.2em" },
            control: {
              width: "1.1em",
            },
          }}
          precision={props.precision}
          step={props.step}
          min={props.min === null ? undefined : props.min[i]}
          max={props.max === null ? undefined : props.max[i]}
          stepHoldDelay={500}
          stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
          disabled={props.disabled}
        />
      ))}
    </Flex>
  );
}

/** GUI input with a label horizontally placed to the left of it. */
export function LabeledInput(props: {
  id: string;
  label: string;
  input: React.ReactNode;
  folderDepth: number;
}) {
  return (
    <Flex align="center">
      <Box
        // The per-layer offset here is just eyeballed.
        w={`${7.25 - props.folderDepth * 0.6375}em`}
        pr="xs"
        sx={{ flexShrink: 0, position: "relative" }}
      >
        <Text
          c="dimmed"
          fz="0.875em"
          fw="450"
          lh="1.375em"
          lts="-0.75px"
          unselectable="off"
          sx={{
            width: "100%",
            boxSizing: "content-box",
          }}
        >
          <label htmlFor={props.id}>{props.label}</label>
        </Text>
      </Box>
      <Box sx={{ flexGrow: 1 }}>{props.input}</Box>
    </Flex>
  );
}