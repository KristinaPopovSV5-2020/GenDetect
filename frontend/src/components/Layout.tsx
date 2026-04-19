import { Box, Stack, Typography } from "@mui/material";
import React, { type JSX, type PropsWithChildren } from "react";

const FOOTER_HEIGHT = 60;

const Layout: React.FC<PropsWithChildren> = ({ children }): JSX.Element => {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        mt: 2.5,
        width: "100%",
        overflowX: "hidden",
      }}
    >
      <Typography variant="h1">Image Authenticity Checker</Typography>
      {/* MAIN CONTENT */}
      <Box
        sx={{
          flex: 1,
          display: "grid",
          paddingBottom: `${FOOTER_HEIGHT}px`,
        }}
      >
        {children}
      </Box>

      {/* FOOTER */}
      <Box
        sx={{
          height: `${FOOTER_HEIGHT}px`,
          zIndex: 10,
        }}
      >
        <Stack
          sx={{
            position: "fixed",
            bottom: 0,
            left: 0,
            width: "100vw",
            marginLeft: "calc(-50vw + 50%)",
            backgroundColor: "secondary.main",
          }}
        >
          <Typography variant="body1" align="center">
            Tinax2 Group
          </Typography>
          <Typography variant="body1" align="center">
            © {new Date().getFullYear()} All rights reserved.
          </Typography>
        </Stack>
      </Box>
    </Box>
  );
};

export default Layout;
