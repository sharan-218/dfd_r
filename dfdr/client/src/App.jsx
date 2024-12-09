import React, { useState } from "react";
import {
  Button,
  Container,
  Typography,
  Box,
  CircularProgress,
  Alert,
  AlertTitle,
  Paper,
} from "@mui/material";
import { CloudUpload } from "@mui/icons-material";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setResult(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        "http://localhost:5000/api/detect_deepfake",
        {
          method: "POST",
          body: formData,
        }
      );
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const data = await response.json();
      setResult({
        type: data.result,
        confidence: data.confidence,
      });
    } catch (error) {
      console.error("Error:", error);
      setResult({
        type: "error",
        message: "An error occurred during processing.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm">
      <Paper elevation={3} sx={{ mt: 4, p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Deepfake Detector
        </Typography>
        <form onSubmit={handleSubmit}>
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 2,
            }}
          >
            <input
              accept="image/*,video/*"
              style={{ display: "none" }}
              id="raised-button-file"
              type="file"
              onChange={handleFileChange}
            />
            <label htmlFor="raised-button-file">
              <Button
                variant="contained"
                component="span"
                startIcon={<CloudUpload />}
              >
                Upload File
              </Button>
            </label>
            {file && (
              <Typography variant="body2">
                Selected file: {file.name}
              </Typography>
            )}
            <Button
              type="submit"
              variant="contained"
              color="primary"
              disabled={!file || loading}
              sx={{ mt: 2 }}
            >
              {loading ? <CircularProgress size={24} /> : "Detect Deepfake"}
            </Button>
          </Box>
        </form>
        {result && (
          <Alert
            severity={
              result.type === "real"
                ? "success"
                : result.type === "deepfake"
                ? "error"
                : "warning"
            }
            sx={{ mt: 2 }}
          >
            <AlertTitle>
              {result.type === "real"
                ? "Authentic"
                : result.type === "deepfake"
                ? "Deepfake"
                : "Error"}
            </AlertTitle>
            {result.type === "real" || result.type === "deepfake" ? (
              <>
                This{" "}
                {result.type === "real"
                  ? "appears to be an authentic"
                  : "may be a deepfake"}{" "}
                image/video.
                <br />
                Confidence: {(result.confidence * 100).toFixed(2)}%
              </>
            ) : (
              result.message
            )}
          </Alert>
        )}
      </Paper>
    </Container>
  );
}

export default App;
