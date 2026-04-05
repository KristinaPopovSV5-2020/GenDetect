import { Box, Button, Typography, Grid } from "@mui/material";
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate";
import "./App.css";
import Layout from "./components/Layout";
import { useState } from "react";

function App() {
  const [image, setImage] = useState<string | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      const file = files[0];
      setImage(URL.createObjectURL(file));
    }
  };

  return (
    <Layout>
      <Grid container spacing={2} sx={{ p: 5 }}>
        <Grid size={{ xs:12, sm:6, md: 4}}>
          <Box
            sx={{
              border: "1px solid #ccc",
              borderRadius: 2,
              padding: 2,
              textAlign: "center",
              height: "350px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexDirection: "column",
            }}
          >
            {image ? (
              <img
                src={image}
                alt="Uploaded"
                style={{ width: "100%", height: "300px", objectFit: "contain" }}
              />
            ) : (
              <>
                <AddPhotoAlternateIcon sx={{ fontSize: 48, mb: 1 }} />
                <p>Upload an image</p>
              </>
            )}
            <Button variant="contained" component="label" sx={{ mt: 1, color: "primary.main" }}>
              Choose Image
              <input
                type="file"
                hidden
                accept="image/*"
                onChange={handleImageChange}
              />
            </Button>
          </Box>
        </Grid>
        <Grid size={{ xs:12, sm:6, md: 4}}>
          <Box sx={{ p: 2 }}>
            <Typography variant="h3">Detection results</Typography>
          </Box>
        </Grid>
      </Grid>
    </Layout>
  );
}

export default App;