import { createTheme } from "@mui/material/styles";
import type { CSSObject } from "@mui/material/styles";

const commonTypography = {
  h1: {
    fontSize: "2.5em",
    lineHeight: 1.1,
    letterSpacing: "1px",
  },
  h3: {
    fontSize: "1.8rem",
  },
  body1: {
    fontSize: "1.0em",
  },
};

const commonComponents = {
  MuiButton: {
    styleOverrides: {
      root: {
        textTransform: "none" as CSSObject["textTransform"],
        fontSize: "1.0rem",
        backgroundColor: "#008080",
        fontWeight: "bold",
      } as CSSObject,
    },
  },
};

export const lightTheme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#000", // Dark text color
    },
    secondary: {
      main: "#008080",
    },
    background: {
      default: "#f8f2f2e7", // White
    },
  },
  typography: commonTypography,
  components: commonComponents,
});

export const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#f8f2f2e7",
    },
    secondary: {
      main: "#008080",
    },
    background: {
      default: "#393939",
    },
  },
  typography: commonTypography,
  components: commonComponents,
});
