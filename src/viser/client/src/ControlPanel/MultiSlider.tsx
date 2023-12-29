import React, { useRef, useState, forwardRef, useEffect } from 'react';
import { useMove, useUncontrolled } from '@mantine/hooks';
import {
  DefaultProps,
  MantineNumberSize,
  MantineColor,
  useMantineTheme,
  useComponentDefaultProps,
  Selectors,
  getSize,
  rem,
} from '@mantine/styles';
import { 
  MantineTransition, 
  Box,
  Transition,
} from '@mantine/core';
import { sizes, useSliderRootStyles, useThumbStyles, useTrackStyles, useMarksStyles } from './MultiSlider.styles';

export function getClientPosition(event: any) {
  if ('TouchEvent' in window && event instanceof window.TouchEvent) {
    const touch = event.touches[0];
    return touch.clientX;
  }

  return event.clientX;
}

interface GetPosition {
  value: number;
  min: number;
  max: number;
}

function getPosition({ value, min, max }: GetPosition) {
  const position = ((value - min) / (max - min)) * 100;
  return Math.min(Math.max(position, 0), 100);
}

interface GetChangeValue {
  value: number;
  containerWidth?: number;
  min: number;
  max: number;
  step: number;
  precision?: number;
}

function getChangeValue({
  value,
  containerWidth,
  min,
  max,
  step,
  precision,
}: GetChangeValue) {
  const left = !containerWidth
    ? value
    : Math.min(Math.max(value, 0), containerWidth) / containerWidth;
  const dx = left * (max - min);
  const nextValue = (dx !== 0 ? Math.round(dx / step) * step : 0) + min;

  const nextValueWithinStep = Math.max(nextValue, min);

  if (precision !== undefined) {
    return Number(nextValueWithinStep.toFixed(precision));
  }

  return nextValueWithinStep;
}

export type SliderRootStylesNames = Selectors<typeof useSliderRootStyles>;

export interface SliderRootProps
  extends DefaultProps<SliderRootStylesNames>,
    React.ComponentPropsWithoutRef<'div'> {
  size: MantineNumberSize;
  children: React.ReactNode;
  disabled: boolean;
  variant: string;
}

export const SliderRoot = forwardRef<HTMLDivElement, SliderRootProps>(
  (
    {
      className,
      size,
      classNames,
      styles,
      disabled,
      unstyled,
      variant,
      ...others
    }: SliderRootProps,
    ref
  ) => {
    const { classes, cx } = useSliderRootStyles((null as unknown) as void,{
      name: 'Slider',
      classNames,
      styles,
      unstyled,
      variant,
      size,
    });
    return <Box {...others} tabIndex={-1} className={cx(classes.root, className)} ref={ref} />;
  }
);

SliderRoot.displayName = '@mantine/core/SliderRoot';


export type ThumbStylesNames = Selectors<typeof useThumbStyles>;

export interface ThumbProps extends DefaultProps<ThumbStylesNames> {
  max: number;
  min: number;
  value: number;
  position: number;
  dragging: boolean;
  color: MantineColor;
  size: MantineNumberSize;
  label: React.ReactNode;
  onKeyDownCapture?(event: React.KeyboardEvent<HTMLDivElement>): void;
  onMouseDown?(event: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>): void;
  labelTransition?: MantineTransition;
  labelTransitionDuration?: number;
  labelTransitionTimingFunction?: string;
  labelAlwaysOn: boolean;
  thumbLabel: string;
  onFocus?(): void;
  onBlur?(): void;
  showLabelOnHover?: boolean;
  isHovered?: boolean;
  children?: React.ReactNode;
  disabled: boolean;
  thumbSize: number;
  variant: string;
}

export const Thumb = forwardRef<HTMLDivElement, ThumbProps>(
  (
    {
      max,
      min,
      value,
      position,
      label,
      dragging,
      onMouseDown,
      onKeyDownCapture,
      color,
      classNames,
      styles,
      size,
      labelTransition,
      labelTransitionDuration,
      labelTransitionTimingFunction,
      labelAlwaysOn,
      thumbLabel,
      onFocus,
      onBlur,
      showLabelOnHover,
      isHovered,
      children = null,
      disabled,
      unstyled,
      thumbSize,
      variant,
    }: ThumbProps,
    ref
  ) => {
    const { classes, cx, theme } = useThumbStyles(
      { color, disabled, thumbSize },
      { name: 'Slider', classNames, styles, unstyled, variant, size }
    );
    const [focused, setFocused] = useState(false);

    const isVisible = labelAlwaysOn || dragging || focused || (showLabelOnHover && isHovered);

    return (
      <Box<'div'>
        tabIndex={0}
        role="slider"
        aria-label={thumbLabel}
        aria-valuemax={max}
        aria-valuemin={min}
        aria-valuenow={value}
        ref={ref}
        className={cx(classes.thumb, { [classes.dragging]: dragging })}
        onFocus={() => {
          setFocused(true);
          typeof onFocus === 'function' && onFocus();
        }}
        onBlur={() => {
          setFocused(false);
          typeof onBlur === 'function' && onBlur();
        }}
        onTouchStart={onMouseDown}
        onMouseDown={onMouseDown}
        onKeyDownCapture={onKeyDownCapture}
        onClick={(event) => event.stopPropagation()}
        style={{ [theme.dir === 'rtl' ? 'right' : 'left']: `${position}%` }}
      >
        {children}
        <Transition
          mounted={(label != null && isVisible) || false}
          duration={labelTransitionDuration}
          transition={labelTransition ?? 'skew-down'}
          timingFunction={labelTransitionTimingFunction || theme.transitionTimingFunction}
        >
          {(transitionStyles) => (
            <div style={transitionStyles} className={classes.label}>
              {label}
            </div>
          )}
        </Transition>
      </Box>
    );
  }
);

Thumb.displayName = '@mantine/core/SliderThumb';

export type MarksStylesNames = Selectors<typeof useMarksStyles>;

export interface MarksProps extends DefaultProps<MarksStylesNames> {
  marks: { value: number; label?: React.ReactNode }[];
  size: MantineNumberSize;
  thumbSize?: number;
  color: MantineColor;
  min: number;
  max: number;
  onChange(value: number): void;
  disabled: boolean;
  variant: string;
}

export function Marks({
  marks,
  color,
  size,
  thumbSize,
  min,
  max,
  classNames,
  styles,
  onChange,
  disabled,
  unstyled,
  variant,
}: MarksProps) {
  const { classes, cx } = useMarksStyles(
    { color, disabled, thumbSize },
    { name: 'Slider', classNames, styles, unstyled, variant, size }
  );

  const items = marks.map((mark, index) => (
    <Box
      className={classes.markWrapper}
      sx={{ left: `${getPosition({ value: mark.value, min, max })}%` }}
      key={index}
    >
      <div
        className={cx(classes.mark, {
          [classes.markFilled]: false,
        })}
      />
      {mark.label && (
        <div
          className={classes.markLabel}
          onMouseDown={(event) => {
            event.stopPropagation();
            !disabled && onChange(mark.value);
          }}
          onTouchStart={(event) => {
            event.stopPropagation();
            !disabled && onChange(mark.value);
          }}
        >
          {mark.label}
        </div>
      )}
    </Box>
  ));

  return <div className={classes.marksContainer}>{items}</div>;
}

Marks.displayName = '@mantine/core/SliderMarks';



export type TrackStylesNames = Selectors<typeof useTrackStyles> | MarksStylesNames;

export interface TrackProps extends DefaultProps<TrackStylesNames> {
  marks: { value: number; label?: React.ReactNode }[];
  size: MantineNumberSize;
  thumbSize?: number;
  radius: MantineNumberSize;
  color: MantineColor;
  min: number;
  max: number;
  children: React.ReactNode;
  onChange(value: number): void;
  disabled: boolean;
  variant: string;
  containerProps?: React.PropsWithRef<React.ComponentProps<'div'>>;
}

export function Track({
  size,
  thumbSize,
  color,
  classNames,
  styles,
  radius,
  children,
  disabled,
  unstyled,
  variant,
  containerProps,
  ...others
}: TrackProps) {
  const { classes } = useTrackStyles(
    { color, radius, disabled, inverted: false },
    { name: 'Slider', classNames, styles, unstyled, variant, size }
  );

  return (
    <>
      <div className={classes.trackContainer} {...containerProps}>
        <div className={classes.track}>
          {children}
        </div>
      </div>

      <Marks
        {...others}
        size={size}
        thumbSize={thumbSize}
        color={color}
        classNames={classNames}
        styles={styles}
        disabled={disabled}
        unstyled={unstyled}
        variant={variant}
      />
    </>
  );
}

Track.displayName = '@mantine/core/SliderTrack';



export type MultiSliderStylesNames =
  | SliderRootStylesNames
  | ThumbStylesNames
  | TrackStylesNames
  | MarksStylesNames;

type Value = number[];

export interface MultiSliderProps
  extends DefaultProps<MultiSliderStylesNames>,
    Omit<React.ComponentPropsWithoutRef<'div'>, 'value' | 'onChange' | 'defaultValue'> {
  variant?: string;

  /** Color from theme.colors */
  color?: MantineColor;

  /** Key of theme.radius or any valid CSS value to set border-radius, theme.defaultRadius by default */
  radius?: MantineNumberSize;

  /** Predefined track and thumb size, number to set sizes */
  size?: MantineNumberSize;

  /** Minimal possible value */
  min?: number;

  /** Maximum possible value */
  max: number;

  /** Minimal range interval */
  minRange: number;

  /** Number by which value will be incremented/decremented with thumb drag and arrows */
  step: number;

  /** Amount of digits after the decimal point */
  precision?: number;

  /** Current value for controlled slider */
  value?: Value;

  /** Default value for uncontrolled slider */
  defaultValue?: Value;

  /** Called each time value changes */
  onChange?(value: Value): void;

  /** Called when user stops dragging slider or changes value with arrows */
  onChangeEnd?(value: Value): void;

  /** Hidden input name, use with uncontrolled variant */
  name?: string;

  /** Marks which will be placed on the track */
  marks: { value: number; label?: React.ReactNode }[];

  /** Function to generate label or any react node to render instead, set to null to disable label */
  label?: React.ReactNode | ((value: number) => React.ReactNode);

  /** Label appear/disappear transition */
  labelTransition?: MantineTransition;

  /** Label appear/disappear transition duration in ms */
  labelTransitionDuration?: number;

  /** Label appear/disappear transition timing function, defaults to theme.transitionRimingFunction */
  labelTransitionTimingFunction?: string;

  /** If true label will be not be hidden when user stops dragging */
  labelAlwaysOn?: boolean;

  /** First thumb aria-label */
  thumbFromLabel?: string;

  /** Second thumb aria-label */
  thumbToLabel?: string;

  /**If true slider label will appear on hover */
  showLabelOnHover?: boolean;

  /** Thumbs children, can be used to add icons */
  thumbChildren?: React.ReactNode | React.ReactNode[] | null;

  /** Disables slider */
  disabled?: boolean;

  /** Thumb width and height */
  thumbSize?: number;

  /** A transformation function, to change the scale of the slider */
  scale?: (value: number) => number;

  fixedEndpoints: boolean;
}

const defaultProps: Partial<MultiSliderProps> = {
  size: 'md',
  radius: 'xl',
  min: 0,
  max: 100,
  minRange: 2,
  step: 1,
  marks: [],
  label: (f) => f,
  labelTransition: 'skew-down',
  labelTransitionDuration: 0,
  labelAlwaysOn: false,
  thumbFromLabel: '',
  thumbChildren: null,
  thumbToLabel: '',
  showLabelOnHover: true,
  disabled: false,
  scale: (v) => v,
  fixedEndpoints: false,
};

export const MultiSlider = forwardRef<HTMLDivElement, MultiSliderProps>((props, ref) => {
  const {
    classNames,
    styles,
    color,
    value,
    onChange,
    onChangeEnd,
    size,
    radius,
    min,
    max,
    minRange,
    step,
    precision,
    defaultValue,
    name,
    marks,
    label,
    labelTransition,
    labelTransitionDuration,
    labelTransitionTimingFunction,
    labelAlwaysOn,
    thumbFromLabel,
    thumbToLabel,
    showLabelOnHover,
    thumbChildren,
    disabled,
    unstyled,
    thumbSize,
    scale,
    variant,
    fixedEndpoints,
    ...others
  } = useComponentDefaultProps('MultiSlider', defaultProps, props) as any;

  const theme = useMantineTheme();
  const [focused, setFocused] = useState(-1);
  const [hovered, setHovered] = useState(false);
  const [_value, setValue] = useUncontrolled<Value>({
    value,
    defaultValue,
    finalValue: [min, max],
    onChange,
  });
  const valueRef = useRef(_value);
  const thumbs = useRef<(HTMLDivElement | null)[]>([]);
  const thumbIndex = useRef<number>(-1);
  const positions = _value.map(x => getPosition({ value: x, min, max}));

  const _setValue = (val: Value) => {
    setValue(val);
    valueRef.current = val;
  };

  useEffect(
    () => {
      if (Array.isArray(value)) {
        valueRef.current = value;
      }
    },
    Array.isArray(value) ? [value[0], value[1]] : [null, null]
  );

  const setRangedValue = (val: number, index: number, triggerChangeEnd: boolean) => {
    const clone: Value = [...valueRef.current];
    clone[index] = val;

    if (index < clone.length - 1) {
      if (val > clone[index + 1] - (minRange - 0.000000001)) {
        clone[index] = Math.max(min, clone[index + 1] - minRange);
      }

      if (val > (max - (minRange - 0.000000001) || min)) {
        clone[index] = valueRef.current[index];
      }
    }

    if (index > 0) {
      if (val < clone[index - 1] + minRange) {
        clone[index] = Math.min(max, clone[index - 1] + minRange);
      }
    }

    if (fixedEndpoints && (index === 0 || index == clone.length - 1)) {
      clone[index] = valueRef.current[index];
    }

    _setValue(clone);

    if (triggerChangeEnd) {
      onChangeEnd?.(valueRef.current);
    }
  };

  const handleChange = (val: number) => {
    if (!disabled) {
      const nextValue = getChangeValue({ value: val, min, max, step, precision });
      setRangedValue(nextValue, thumbIndex.current, false);
    }
  };

  const { ref: container, active } = useMove(
    ({ x }) => handleChange(x),
    { onScrubEnd: () => onChangeEnd?.(valueRef.current) },
    theme.dir
  );

  function handleThumbMouseDown(index: number) {
    thumbIndex.current = index;
  }

  const handleTrackMouseDownCapture = (
    event: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>
  ) => {
    container.current.focus();
    const rect = container.current.getBoundingClientRect();
    const changePosition = getClientPosition(event.nativeEvent);
    const changeValue = getChangeValue({
      value: changePosition - rect.left,
      max,
      min,
      step,
      containerWidth: rect.width,
    });

    const _nearestHandle = _value.map((v) => Math.abs(v - changeValue)).indexOf(Math.min(..._value.map((v) => Math.abs(v - changeValue))));

    thumbIndex.current = _nearestHandle;
  };

  const getFocusedThumbIndex = () => {
    if (focused !== 1 && focused !== 0) {
      setFocused(0);
      return 0;
    }

    return focused;
  };

  const handleTrackKeydownCapture = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (!disabled) {
      switch (event.key) {
        case 'ArrowUp': {
          event.preventDefault();
          const focusedIndex = getFocusedThumbIndex();
          thumbs.current[focusedIndex]?.focus();
          setRangedValue(
            Math.min(Math.max(valueRef.current[focusedIndex] + step, min), max),
            focusedIndex,
            true
          );
          break;
        }
        case 'ArrowRight': {
          event.preventDefault();
          const focusedIndex = getFocusedThumbIndex();
          thumbs.current[focusedIndex]?.focus();
          setRangedValue(
            Math.min(
              Math.max(
                theme.dir === 'rtl'
                  ? valueRef.current[focusedIndex] - step
                  : valueRef.current[focusedIndex] + step,
                min
              ),
              max
            ),
            focusedIndex,
            true
          );
          break;
        }

        case 'ArrowDown': {
          event.preventDefault();
          const focusedIndex = getFocusedThumbIndex();
          thumbs.current[focusedIndex]?.focus();
          setRangedValue(
            Math.min(Math.max(valueRef.current[focusedIndex] - step, min), max),
            focusedIndex,
            true
          );
          break;
        }
        case 'ArrowLeft': {
          event.preventDefault();
          const focusedIndex = getFocusedThumbIndex();
          thumbs.current[focusedIndex]?.focus();
          setRangedValue(
            Math.min(
              Math.max(
                theme.dir === 'rtl'
                  ? valueRef.current[focusedIndex] + step
                  : valueRef.current[focusedIndex] - step,
                min
              ),
              max
            ),
            focusedIndex,
            true
          );
          break;
        }

        default: {
          break;
        }
      }
    }
  };

  const sharedThumbProps = {
    max,
    min,
    color,
    size,
    labelTransition,
    labelTransitionDuration,
    labelTransitionTimingFunction,
    labelAlwaysOn,
    onBlur: () => setFocused(-1),
    classNames,
    styles,
  };

  const hasArrayThumbChildren = Array.isArray(thumbChildren);

  return (
    <SliderRoot
      {...others}
      size={size}
      ref={ref}
      styles={styles}
      classNames={classNames}
      disabled={disabled}
      unstyled={unstyled}
      variant={variant}
    >
      <Track
        marks={marks}
        size={size}
        thumbSize={thumbSize}
        radius={radius}
        color={color}
        min={min}
        max={max}
        styles={styles}
        classNames={classNames}
        onChange={(val) => {
          const nearestValue = Math.abs(_value[0] - val) > Math.abs(_value[1] - val) ? 1 : 0;
          const clone: Value = [..._value];
          clone[nearestValue] = val;
          _setValue(clone);
        }}
        disabled={disabled}
        unstyled={unstyled}
        variant={variant}
        containerProps={{
          ref: container,
          onMouseEnter: showLabelOnHover ? () => setHovered(true) : undefined,
          onMouseLeave: showLabelOnHover ? () => setHovered(false) : undefined,
          onTouchStartCapture: handleTrackMouseDownCapture,
          onTouchEndCapture: () => {
            thumbIndex.current = -1;
          },
          onMouseDownCapture: handleTrackMouseDownCapture,
          onMouseUpCapture: () => {
            thumbIndex.current = -1;
          },
          onKeyDownCapture: handleTrackKeydownCapture,
        }}
      >{_value.map((value, index) => (
        <Thumb
          {...sharedThumbProps}
          value={scale(value)}
          key={index}
          position={positions[index]}
          dragging={active}
          label={typeof label === 'function' ? label(scale(value)) : label}
          ref={(node) => {
            thumbs.current[index] = node;
          }}
          thumbLabel={thumbFromLabel}
          onMouseDown={() => handleThumbMouseDown(index)}
          onFocus={() => setFocused(index)}
          showLabelOnHover={showLabelOnHover}
          isHovered={hovered}
          disabled={disabled}
          unstyled={unstyled}
          thumbSize={thumbSize}
          variant={variant}
        >
          {hasArrayThumbChildren ? thumbChildren[index] : thumbChildren}
        </Thumb>))}
      </Track>
      {_value.map((value, index) => (
        <input type="hidden" name={`${name}[]`} key={index} value={value} />
      ))}
    </SliderRoot>
  );
});

MultiSlider.displayName = 'MultiSlider';