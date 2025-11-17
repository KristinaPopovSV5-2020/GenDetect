import { Box, Stack, Typography } from "@mui/material";
import React, { type JSX, type PropsWithChildren } from "react";

const FOOTER_HEIGHT = 5;

const Layout: React.FC<PropsWithChildren> = ({ children }): JSX.Element => {
  return (
    <Box
      display="flex"
      flexDirection="column"
      sx={{
        minHeight: "100vh",
        maxWidth: "100vw",
        overflowX: "hidden",
      }}
    >
      <Typography variant="h1">Image Authenticity Checker</Typography>
      {/* MAIN CONTENT */}
      <Box
        sx={{
          flex: 1,
          display: "grid",
          marginTop: 20,
          gridTemplateColumns: "1fr 1fr",
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
            width: "100%",
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
