import { createStyles, rem } from '@mantine/styles';
import { MantineColor, getSize, MantineNumberSize } from '@mantine/styles';


export const sizes = {
  xs: rem(4),
  sm: rem(6),
  md: rem(8),
  lg: rem(10),
  xl: rem(12),
};

export const useSliderRootStyles = createStyles((theme) => ({
  root: {
    ...theme.fn.fontStyles(),
    WebkitTapHighlightColor: 'transparent',
    outline: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    touchAction: 'none',
    position: 'relative',
  },
}));


interface ThumbStyles {
  color: MantineColor;
  disabled: boolean;
  thumbSize: number | string;
}

export const useThumbStyles = createStyles((theme, { color, disabled, thumbSize }: ThumbStyles, { size }) => ({
  label: {
    position: 'absolute',
    top: rem(-36),
    backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[4] : theme.colors.gray[9],
    fontSize: theme.fontSizes.xs,
    color: theme.white,
    padding: `calc(${theme.spacing.xs} / 2)`,
    borderRadius: theme.radius.sm,
    whiteSpace: 'nowrap',
    pointerEvents: 'none',
    userSelect: 'none',
    touchAction: 'none',
  },

  thumb: {
    ...theme.fn.focusStyles(),
    boxSizing: 'border-box',
    position: 'absolute',
    display: 'flex',
    height: thumbSize ? rem(thumbSize) : `calc(${getSize({ sizes, size })} * 2)`,
    width: thumbSize ? rem(thumbSize) : `calc(${getSize({ sizes, size })} * 2)`,
    backgroundColor:
      disabled ? 
        theme.colorScheme === 'dark' ?
          theme.colors.dark[3] :
          theme.colors.gray[4] :
        theme.colorScheme === 'dark'
          ? theme.fn.themeColor(color, theme.fn.primaryShade())
          : theme.white,
    border: `${rem(4)} solid ${
      disabled ? 
        theme.colorScheme === 'dark' ?
          theme.colors.dark[3] :
          theme.colors.gray[4] :
        theme.colorScheme === 'dark'
          ? theme.white
          : theme.fn.themeColor(color, theme.fn.primaryShade())
    }`,
    color:
      theme.colorScheme === 'dark'
        ? theme.white
        : theme.fn.themeColor(color, theme.fn.primaryShade()),
    transform: 'translate(-50%, -50%)',
    top: '50%',
    cursor: disabled ? 'not-allowed' : 'pointer',
    borderRadius: 1000,
    alignItems: 'center',
    justifyContent: 'center',
    transitionDuration: '100ms',
    transitionProperty: 'box-shadow, transform',
    transitionTimingFunction: theme.transitionTimingFunction,
    zIndex: 3,
    userSelect: 'none',
    touchAction: 'none',
  },

  dragging: {
    transform: 'translate(-50%, -50%) scale(1.05)',
    boxShadow: theme.shadows.sm,
  },
}));


interface TrackStyles {
    radius: MantineNumberSize;
    color: MantineColor;
    disabled: boolean;
    inverted: boolean;
    thumbSize?: number;
  }

  export const useTrackStyles = createStyles(
    (theme, { radius, color, disabled, inverted, thumbSize }: TrackStyles, { size }) => ({
      trackContainer: {
        display: 'flex',
        alignItems: 'center',
        width: '100%',
        height: `calc(${getSize({ sizes, size })} * 2)`,
        cursor: 'pointer',

        '&:has(~ input:disabled)': {
          '&': {
            pointerEvents: 'none',
          },

          '& .mantine-Slider-thumb': {
            display: 'none',
          },

          '& .mantine-Slider-track::before': {
            content: '""',
            backgroundColor: inverted
              ? theme.colorScheme === 'dark'
                ? theme.colors.dark[3]
                : theme.colors.gray[4]
              : theme.colorScheme === 'dark'
              ? theme.colors.dark[4]
              : theme.colors.gray[2],
          },

          '& .mantine-Slider-bar': {
            backgroundColor: inverted
              ? theme.colorScheme === 'dark'
                ? theme.colors.dark[4]
                : theme.colors.gray[2]
              : theme.colorScheme === 'dark'
              ? theme.colors.dark[3]
              : theme.colors.gray[4],
          },
        },
      },

      track: {
        position: 'relative',
        height: getSize({ sizes, size }),
        width: '100%',
        marginRight: thumbSize ? rem(thumbSize / 2) : getSize({ size, sizes }),
        marginLeft: thumbSize ? rem(thumbSize / 2) : getSize({ size, sizes }),

        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          bottom: 0,
          borderRadius: theme.fn.radius(radius),
          right: `calc(${thumbSize ? rem(thumbSize / 2) : getSize({ size, sizes })} * -1)`,
          left: `calc(${thumbSize ? rem(thumbSize / 2) : getSize({ size, sizes })} * -1)`,
          backgroundColor: inverted
            ? disabled
              ? theme.colorScheme === 'dark'
                ? theme.colors.dark[3]
                : theme.colors.gray[4]
              : theme.fn.variant({ variant: 'filled', color }).background
            : theme.colorScheme === 'dark'
            ? theme.colors.dark[4]
            : theme.colors.gray[2],
          zIndex: 0,
        },
      },

      bar: {
        position: 'absolute',
        zIndex: 1,
        top: 0,
        bottom: 0,
        backgroundColor: inverted
          ? theme.colorScheme === 'dark'
            ? theme.colors.dark[4]
            : theme.colors.gray[2]
          : disabled
          ? theme.colorScheme === 'dark'
            ? theme.colors.dark[3]
            : theme.colors.gray[4]
          : theme.fn.variant({ variant: 'filled', color }).background,
        borderRadius: theme.fn.radius(radius),
      },
    })
  );

  interface MarksStyles {
  color: MantineColor;
  disabled: boolean;
  thumbSize?: number;
}

export const useMarksStyles = createStyles((theme, { color, disabled, thumbSize }: MarksStyles, { size }) => ({
  marksContainer: {
    position: 'absolute',
    right: thumbSize ? rem(thumbSize / 2) : getSize({ sizes, size }),
    left: thumbSize ? rem(thumbSize / 2) : getSize({ sizes, size }),

    '&:has(~ input:disabled)': {
      '& .mantine-Slider-markFilled': {
        border: `${rem(2)} solid ${
          theme.colorScheme === 'dark' ? theme.colors.dark[4] : theme.colors.gray[2]
        }`,
        borderColor: theme.colorScheme === 'dark' ? theme.colors.dark[3] : theme.colors.gray[4],
      },
    },
  },

  markWrapper: {
    position: 'absolute',
    top: `calc(${rem(getSize({ sizes, size }))} / 2)`,
    zIndex: 2,
    height: 0,
  },

  mark: {
    boxSizing: 'border-box',
    border: `${rem(2)} solid ${
      theme.colorScheme === 'dark' ? theme.colors.dark[4] : theme.colors.gray[2]
    }`,
    height: getSize({ sizes, size }),
    width: getSize({ sizes, size }),
    borderRadius: 1000,
    transform: `translateX(calc(-${getSize({ sizes, size })} / 2))`,
    backgroundColor: theme.white,
    pointerEvents: 'none',
  },

  markFilled: {
    borderColor: disabled
      ? theme.colorScheme === 'dark'
        ? theme.colors.dark[3]
        : theme.colors.gray[4]
      : theme.fn.variant({ variant: 'filled', color }).background,
  },

  markLabel: {
    transform: `translate(-50%, calc(${theme.spacing.xs} / 2))`,
    fontSize: theme.fontSizes.sm,
    color: theme.colorScheme === 'dark' ? theme.colors.dark[2] : theme.colors.gray[6],
    whiteSpace: 'nowrap',
    cursor: 'pointer',
    userSelect: 'none',
  },
}));
