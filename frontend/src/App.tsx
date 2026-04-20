import { Box, Button, Grid, Typography } from "@mui/material";
import "./App.css";
import Layout from "./components/Layout";
import { useState, useRef } from "react";
import { predictImage, getGradCAM } from "./api";
import type { PredictResponse } from "./types";
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate";
import DetectionTable from "./components/DetectionTable";

interface Results {
  resnet: PredictResponse;
  vit: PredictResponse;
  gradcamResnet: string;
  gradcamVit: string | null;
  attentionVit: string | null;
}

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<File | null>(null);

  const handleImageChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    fileRef.current = file;
    setImage(URL.createObjectURL(file));
    setResults(null);
    setError(null);
    setLoading(true);

    try {
      const [resnet, vit, gradcamResnet, gradcamVit, attentionVit] = await Promise.all([
        predictImage(file, "resnet"),
        predictImage(file, "vit"),
        getGradCAM(file, "resnet_gradcam"),
        getGradCAM(file, "vit_gradcam"),
        getGradCAM(file, "vit_attention")
      ]);
      setResults({ resnet, vit, gradcamResnet, gradcamVit, attentionVit});
    } catch (err) {
      setError("Error processing image. Make sure the server is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <Box sx={{ p: { xs: 2, md: 4 }, width: "100%" }}>
        <Grid container sx={{ alignItems: "flex-start" }} spacing={2}>
          {/* Upload */}
          <Grid size={{ xs: 12, md: 6 }}>
            <Box
              sx={{
                border: "0.5px solid",
                borderColor: "divider",
                borderRadius: 2,
                p: 1.5,
                bgcolor: "background.paper",
                textAlign: "center",
              }}
            >
              <Box
                sx={{
                  border: "1.5px dashed",
                  borderColor: "divider",
                  borderRadius: 2,
                  height: 300,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 1,
                  overflow: "hidden",
                }}
              >
                {image ? (
                  <img
                    src={image}
                    alt="Uploaded"
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                    }}
                  />
                ) : (
                  <>
                    <AddPhotoAlternateIcon sx={{ fontSize: 52 }} />
                    <span style={{ fontSize: 15 }}>Upload an image</span>
                  </>
                )}
              </Box>
              <Button
                variant="contained"
                component="label"
                sx={{ mt: 1.5, color: "primary.main" }}
              >
                {loading ? "Processing..." : "Choose image"}
                <input
                  type="file"
                  hidden
                  accept="image/*"
                  onChange={handleImageChange}
                />
              </Button>
              {error && (
                <Box sx={{ mt: 1, fontSize: 13, color: "error.main" }}>
                  {error}
                </Box>
              )}
            </Box>
          </Grid>

          {/* Results table */}
          <Grid size={{ xs: 12, md: 6 }} sx={{ display: "flex" }}>
            <Box
              sx={{
                height: "100%",
                display: "flex",
                width: "100%",
                flexDirection: "column",
              }}
            >
              <DetectionTable results={results} loading={loading} />
            </Box>
          </Grid>

          {results && (
            <>
              {[
                { title: "Grad-CAM — ResNet50", src: results.gradcamResnet },
                { title: "Grad-CAM — ViT", src: results.gradcamVit },
                { title: "Attention-Rollout — ViT", src: results.attentionVit },
              ].map(({ title, src }) => (
                <Grid key={title} size={{ xs: 12, sm: 4 }}>
                  <Box
                    sx={{
                      border: "0.5px solid",
                      borderColor: "divider",
                      borderRadius: 2,
                      bgcolor: "background.paper",
                      p: 1.5,
                      height: "50vh",
                      display: "flex",
                      flexDirection: "column",
                    }}
                  >
                    <Typography
                      variant="subtitle1"
                      sx={{ fontWeight: 500, color: "text.secondary", mb: 1 }}
                    >
                      {title}
                    </Typography>

                    {src ? (
                      <Box sx={{ flex: 1, overflow: "hidden" }}>
                        <img
                          src={src}
                          alt={title}
                          style={{
                            width: "100%",
                            height: "100%",
                            borderRadius: 8,
                            display: "block",
                            objectFit: "contain",
                          }}
                        />
                      </Box>
                    ) : (
                      <Box
                        sx={{
                          flex: 1,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          bgcolor: "grey.50",
                          borderRadius: 2,
                          color: "text.disabled",
                          fontSize: 13,
                        }}
                      >
                        Not available yet
                      </Box>
                    )}
                  </Box>
                </Grid>
              ))}
            </>
          )}
        </Grid>
      </Box>
    </Layout>
  );
}
