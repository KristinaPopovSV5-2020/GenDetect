import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import type { PredictResponse } from "../types";

interface Results {
  resnet: PredictResponse;
  vit: PredictResponse;
}

function VerdictBadge({ prediction }: { prediction: number }) {
  const isReal = prediction === 0;
  return (
    <span
      style={{
        background: isReal ? "#d1fae5" : "#fee2e2",
        color: isReal ? "#065f46" : "#991b1b",
        fontSize: 14,
        fontWeight: 600,
        padding: "2px 8px",
        borderRadius: 5,
        letterSpacing: "0.05em",
      }}
    >
      {isReal ? "REAL" : "FAKE"}
    </span>
  );
}

function getFakeProb(r: PredictResponse) {
  return r.prediction === 1 ? r.probability : 1 - r.probability;
}
function getRealProb(r: PredictResponse) {
  return r.prediction === 0 ? r.probability : 1 - r.probability;
}

interface DetectionTableProps {
  results: Results | null;
  loading: boolean;
}

export default function DetectionTable({
  results,
  loading,
}: DetectionTableProps) {
  const rows = results
    ? [
        {
          label: "Verdict",
          resnet: <VerdictBadge prediction={results.resnet.prediction} />,
          vit: <VerdictBadge prediction={results.vit.prediction} />,
        },
        {
          label: "Fake prob.",
          resnet: (
            <span
              style={{ fontSize: "1.3rem", fontWeight: 500, color: "#b91c1c" }}
            >
              {Math.round(getFakeProb(results.resnet) * 100)}%
            </span>
          ),
          vit: (
            <span
              style={{ fontSize: "1.3rem", fontWeight: 500, color: "#b91c1c" }}
            >
              {Math.round(getFakeProb(results.vit) * 100)}%
            </span>
          ),
        },
        {
          label: "Real prob.",
          resnet: (
            <span
              style={{ fontSize: "1.3rem", fontWeight: 500, color: "#00ff99" }}
            >
              {Math.round(getRealProb(results.resnet) * 100)}%
            </span>
          ),
          vit: (
            <span
              style={{ fontSize: "1.3rem", fontWeight: 500, color: "#00ff99" }}
            >
              {Math.round(getRealProb(results.vit) * 100)}%
            </span>
          ),
        },
      ]
    : [];

  return (
    <Box
      sx={{
        borderColor: "divider",
        borderRadius: 2,
        bgcolor: "background.paper",
        overflow: "hidden",
        flexDirection: "column",
        height: "100%",
      }}
    >
      <Box
        sx={{
          px: 1.5,
          py: 2,
          borderBottom: "0.5px solid",
          borderColor: "divider",
        }}
      >
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Detection Results
        </Typography>
      </Box>

      <TableContainer>
        <Table>
          <TableHead>
            <TableRow sx={{ bgcolor: "action.hover" }}>
              {["Metric", "ResNet50", "ViT"].map((h) => (
                <TableCell
                  key={h}
                  align={h === "Metric" ? "left" : "center"}
                  sx={{
                    py: 2,
                    color: "text.secondary",
                    borderBottom: "0.5px solid",
                    borderColor: "divider",
                    fontSize: 18,
                  }}
                >
                  {h}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>

          <TableBody>
            {results ? (
              rows.map((row, i) => (
                <TableRow
                  key={i}
                  sx={{
                    "&:last-child td": { border: 0 },
                    "&:not(:last-child) td": {
                      borderBottom: "0.5px solid",
                      borderColor: "divider",
                    },
                  }}
                >
                  <TableCell sx={{ color: "text.secondary", py: 2.5 }}>
                    {row.label}
                  </TableCell>
                  <TableCell align="center" sx={{ py: 2.5 }}>
                    {row.resnet}
                  </TableCell>
                  <TableCell align="center" sx={{ py: 2.5 }}>
                    {row.vit}
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={3}
                  align="center"
                  sx={{
                    py: 3,
                    color: "text.disabled",
                    fontSize: 16,
                    border: 0,
                  }}
                >
                  {loading
                    ? "Analyzing image..."
                    : "Upload an image to see results"}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
