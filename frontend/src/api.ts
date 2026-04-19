import type { PredictResponse } from "./types";

const BASE_URL = "http://localhost:8000";

export async function predictImage(
  file: File,
  modelName: "resnet" | "vit",
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", modelName);

  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getGradCAM(file: File, model_name: string): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", model_name);

  const res = await fetch(`${BASE_URL}/xai`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
