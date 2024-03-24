import {
  Checkbox,
  ColorInput,
  Select,
  TextInput,
  NumberInput,
  Paper,
  ActionIcon,
  Button,
  createTheme,
} from "@mantine/core";
import {themeToVars} from "@mantine/vanilla-extract";

export const theme = createTheme({
  fontFamily: "Inter",
  components: {
    Checkbox: Checkbox.extend({
      defaultProps: {
        radius: "xs",
      },
    }),
    ColorInput: ColorInput.extend({
      defaultProps: {
        radius: "xs",
      },
    }),
    Select: Select.extend({
      defaultProps: {
        radius: "sm",
      },
    }),
    TextInput: TextInput.extend({
      defaultProps: {
        radius: "xs",
      },
    }),
    NumberInput: NumberInput.extend({
      defaultProps: {
        radius: "xs",
      },
    }),
    Paper: Paper.extend({
      defaultProps: {
        radius: "xs",
        shadow: "0",
      },
    }),
    ActionIcon: ActionIcon.extend({
      defaultProps: {
        variant: "subtle",
        color: "gray",
        radius: "xs",
      },
    }),
    Button: Button.extend({
      defaultProps: {
        radius: "xs",
        fw: 450,
      },
    }),
  },
});

export const vars = themeToVars(theme);
