import React, { useCallback, useRef, useState } from "react";
import { clamp, useMergedRef, useMove, useUncontrolled } from "@mantine/hooks";
import {
  BoxProps,
  createVarsResolver,
  ElementProps,
  factory,
  Factory,
  getRadius,
  getSize,
  getThemeColor,
  MantineColor,
  MantineRadius,
  MantineSize,
  rem,
  StylesApiProps,
  useDirection,
  useProps,
  useStyles,
  TransitionOverride,
} from "@mantine/core";
import {
  SliderCssVariables,
  SliderProvider,
  SliderStylesNames,
} from "../Slider.context";
import { SliderRoot } from "../SliderRoot/SliderRoot";
import { Thumb } from "../Thumb/Thumb";
import { Track } from "../Track/Track";
import { getChangeValue } from "../utils/get-change-value/get-change-value";
import { getPosition } from "../utils/get-position/get-position";
import { getPrecision } from "../utils/get-precision/get-precision";
import classes from "../Slider.module.css";

export interface SliderProps
  extends BoxProps,
    StylesApiProps<SliderFactory>,
    ElementProps<"div", "onChange" | "defaultValue"> {
  /** Key of `theme.colors` or any valid CSS color, controls color of track and thumb, `theme.primaryColor` by default */
  color?: MantineColor;

  /** Key of `theme.radius` or any valid CSS value to set `border-radius`, numbers are converted to rem, `'xl'` by default */
  radius?: MantineRadius;

  /** Controls size of the track, `'md'` by default */
  size?: MantineSize | (string & NonNullable<unknown>) | number;

  /** Minimal possible value, `0` by default */
  min?: number;

  /** Maximum possible value, `100` by default */
  max?: number;

  /** Number by which value will be incremented/decremented with thumb drag and arrows, `1` by default */
  step?: number;

  /** Number of significant digits after the decimal point */
  precision?: number;

  fixedEndpoints: boolean;
  minRange?: number;

  /** Controlled component value */
  value?: number[];

  /** Uncontrolled component default value */
  defaultValue?: number[];

  /** Called when value changes */
  onChange?: (value: number[]) => void;

  /** Called when user stops dragging slider or changes value with arrows */
  onChangeEnd?: (value: number[]) => void;

  /** Hidden input name, use with uncontrolled component */
  name?: string;

  /** Marks displayed on the track */
  marks?: { value: number; label?: React.ReactNode }[];

  /** Function to generate label or any react node to render instead, set to null to disable label */
  label?: React.ReactNode | ((value: number) => React.ReactNode);

  /** Props passed down to the `Transition` component, `{ transition: 'fade', duration: 0 }` by default */
  labelTransitionProps?: TransitionOverride;

  /** Determines whether the label should be visible when the slider is not being dragged or hovered, `false` by default */
  labelAlwaysOn?: boolean;

  /** Thumb `aria-label` */
  thumbLabel?: string;

  /** Determines whether the label should be displayed when the slider is hovered, `true` by default */
  showLabelOnHover?: boolean;

  /** Content rendered inside thumb */
  thumbChildren?: React.ReactNode;

  /** Disables slider */
  disabled?: boolean;

  /** Thumb `width` and `height`, by default value is computed based on `size` prop */
  thumbSize?: number | string;

  /** A transformation function to change the scale of the slider */
  scale?: (value: number) => number;

  /** Determines whether track value representation should be inverted, `false` by default */
  inverted?: boolean;

  /** Props passed down to the hidden input */
  hiddenInputProps?: React.ComponentPropsWithoutRef<"input">;
}

export type SliderFactory = Factory<{
  props: SliderProps;
  ref: HTMLDivElement;
  stylesNames: SliderStylesNames;
  vars: SliderCssVariables;
}>;

const defaultProps: Partial<SliderProps> = {
  radius: "xl",
  min: 0,
  max: 100,
  step: 1,
  fixedEndpoints: true,
  marks: [],
  label: (f) => f,
  labelTransitionProps: { transition: "fade", duration: 0 },
  labelAlwaysOn: false,
  thumbLabel: "",
  showLabelOnHover: true,
  disabled: false,
  scale: (v) => v,
};

const varsResolver = createVarsResolver<SliderFactory>(
  (theme, { size, color, thumbSize, radius }) => ({
    root: {
      "--slider-size": getSize(size, "slider-size"),
      "--slider-color": color ? getThemeColor(color, theme) : undefined,
      "--slider-radius": radius === undefined ? undefined : getRadius(radius),
      "--slider-thumb-size":
        thumbSize !== undefined
          ? rem(thumbSize)
          : "calc(var(--slider-size) * 2)",
    },
  }),
);

export const MultiSlider = factory<SliderFactory>((_props, ref) => {
  const props = useProps("MultiSlider", defaultProps, _props);
  const {
    classNames,
    styles,
    value,
    onChange,
    onChangeEnd,
    size,
    min,
    max,
    step,
    fixedEndpoints,
    minRange,
    precision: _precision,
    defaultValue,
    name,
    marks,
    label,
    labelTransitionProps,
    labelAlwaysOn,
    thumbLabel,
    showLabelOnHover,
    thumbChildren,
    disabled,
    unstyled,
    scale,
    inverted,
    className,
    style,
    vars,
    // hiddenInputProps,
    ...others
  } = props;

  const getStyles = useStyles<SliderFactory>({
    name: "MultiSlider",
    props,
    classes,
    classNames,
    className,
    styles,
    style,
    vars,
    varsResolver,
    unstyled,
  });

  const { dir } = useDirection();
  const [hovered, setHovered] = useState(false);
  const [_value, setValue] = useUncontrolled({
    value: value === undefined ? value : value.map((x) => clamp(x, min!, max!)),
    defaultValue:
      defaultValue === undefined
        ? defaultValue
        : defaultValue.map((x) => clamp(x, min!, max!)),
    finalValue: [clamp(0, min!, max!)],
    onChange,
  });

  const valueRef = useRef(_value);
  const root = useRef<HTMLDivElement>();
  const thumbs = useRef<(HTMLDivElement | null)[]>([]);
  const thumbIndex = useRef<number>(-1);
  const positions = _value.map((x) =>
    getPosition({ value: x, min: min!, max: max! }),
  );
  const precision = _precision ?? getPrecision(step!);

  valueRef.current = _value;

  const setRangedValue = (
    val: number,
    thumbIndex: number,
    triggerChangeEnd: boolean,
  ) => {
    const clone: number[] = [...valueRef.current];
    clone[thumbIndex] = val;

    const _minRange = minRange || step!;
    if (thumbIndex < clone.length - 1) {
      if (val > clone[thumbIndex + 1] - (_minRange - 0.000000001)) {
        clone[thumbIndex] = Math.max(min!, clone[thumbIndex + 1] - _minRange);
      }

      if (val > (max! - (_minRange - 0.000000001) || min!)) {
        clone[thumbIndex] = valueRef.current[thumbIndex];
      }
    }

    if (thumbIndex > 0) {
      if (val < clone[thumbIndex - 1] + _minRange) {
        clone[thumbIndex] = Math.min(max!, clone[thumbIndex - 1] + _minRange);
      }
    }

    // const fixedEndpoints = false;
    if (
      fixedEndpoints &&
      (thumbIndex === 0 || thumbIndex == clone.length - 1)
    ) {
      clone[thumbIndex] = valueRef.current[thumbIndex];
    }

    setValue(clone);
    valueRef.current = clone;

    if (triggerChangeEnd) {
      onChangeEnd?.(valueRef.current);
    }
  };

  const handleChange = useCallback(
    ({ x }: { x: number }) => {
      if (!disabled) {
        const nextValue = getChangeValue({
          value: x,
          min: min!,
          max: max!,
          step: step!,
          precision,
        });
        setRangedValue(nextValue, thumbIndex.current, false);
      }
    },
    [disabled, min, max, step, precision, setValue],
  );

  const { ref: container, active } = useMove(
    handleChange,
    { onScrubEnd: () => onChangeEnd?.(valueRef.current) },
    dir,
  );

  function getClientPosition(event: any) {
    if ("TouchEvent" in window && event instanceof window.TouchEvent) {
      const touch = event.touches[0];
      return touch.clientX;
    }

    return event.clientX;
  }
  const handleTrackMouseDownCapture = (
    event: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>,
  ) => {
    container.current!.focus();
    const rect = container.current!.getBoundingClientRect();
    const changePosition = getClientPosition(event.nativeEvent);
    const changeValue = getChangeValue({
      value: changePosition - rect.left,
      max: max!,
      min: min!,
      step: step!,
      containerWidth: rect.width,
    });

    const _nearestHandle = _value
      .map((v) => Math.abs(v - changeValue))
      .indexOf(Math.min(..._value.map((v) => Math.abs(v - changeValue))));

    thumbIndex.current = _nearestHandle;
  };

  const handleTrackKeydownCapture = (
    event: React.KeyboardEvent<HTMLDivElement>,
  ) => {
    if (!disabled) {
      const focusedIndex = thumbIndex.current;
      switch (event.key) {
        case "ArrowUp": {
          event.preventDefault();
          thumbs.current[focusedIndex]!.focus();
          setRangedValue(
            Math.min(
              Math.max(valueRef.current[focusedIndex] + step!, min!),
              max!,
            ),
            focusedIndex,
            true,
          );
          break;
        }
        case "ArrowRight": {
          event.preventDefault();
          thumbs.current[focusedIndex]?.focus();
          setRangedValue(
            Math.min(
              Math.max(
                dir === "rtl"
                  ? valueRef.current[focusedIndex] - step!
                  : valueRef.current[focusedIndex] + step!,
                min!,
              ),
              max!,
            ),
            focusedIndex,
            true,
          );
          break;
        }

        case "ArrowDown": {
          event.preventDefault();
          thumbs.current[focusedIndex]?.focus();
          setRangedValue(
            Math.min(
              Math.max(valueRef.current[focusedIndex] - step!, min!),
              max!,
            ),
            focusedIndex,
            true,
          );
          break;
        }
        case "ArrowLeft": {
          event.preventDefault();
          thumbs.current[focusedIndex]?.focus();
          setRangedValue(
            Math.min(
              Math.max(
                dir === "rtl"
                  ? valueRef.current[focusedIndex] + step!
                  : valueRef.current[focusedIndex] - step!,
                min!,
              ),
              max!,
            ),
            focusedIndex,
            true,
          );
          break;
        }

        case "Home": {
          event.preventDefault();
          thumbs.current[focusedIndex]?.focus();
          setRangedValue(min!, focusedIndex, true);
          break;
        }

        case "End": {
          event.preventDefault();
          thumbs.current[focusedIndex]?.focus();
          setRangedValue(max!, focusedIndex, true);
          break;
        }

        default: {
          break;
        }
      }
    }
  };

  return (
    <SliderProvider value={{ getStyles }}>
      <SliderRoot
        {...others}
        ref={useMergedRef(ref, root)}
        onKeyDownCapture={handleTrackKeydownCapture}
        onMouseDownCapture={() => root.current?.focus()}
        size={size!}
        disabled={disabled}
      >
        <Track
          inverted={inverted}
          offset={0}
          filled={0}
          value={0}
          marks={marks}
          min={min!}
          max={max!}
          disabled={disabled}
          containerProps={{
            ref: container as any,
            onMouseEnter: showLabelOnHover ? () => setHovered(true) : undefined,
            onMouseLeave: showLabelOnHover
              ? () => setHovered(false)
              : undefined,
            onTouchStartCapture: handleTrackMouseDownCapture,
            onTouchEndCapture: () => {
              thumbIndex.current = -1;
            },
            onMouseDownCapture: handleTrackMouseDownCapture,
            onMouseUpCapture: () => {
              thumbIndex.current = -1;
            },
          }}
        >
          {_value.map((value, index) => (
            <Thumb
              key={index}
              max={max!}
              min={min!}
              value={scale!(value)}
              position={positions[index]}
              dragging={active}
              draggingThisThumb={active && thumbIndex.current === index}
              label={typeof label === "function" ? label(scale!(value)) : label}
              ref={(node) => {
                thumbs.current[index] = node;
              }}
              labelTransitionProps={labelTransitionProps}
              labelAlwaysOn={labelAlwaysOn}
              thumbLabel={thumbLabel}
              showLabelOnHover={showLabelOnHover}
              isHovered={hovered}
              disabled={disabled}
            >
              {thumbChildren}
            </Thumb>
          ))}
        </Track>
        {_value.map((value, index) => (
          <input type="hidden" name={`${name}[]`} key={index} value={value} />
        ))}
      </SliderRoot>
    </SliderProvider>
  );
});

MultiSlider.classes = classes;
MultiSlider.displayName = "MultiSlider";
