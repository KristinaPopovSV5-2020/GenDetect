export interface PredictRequest {
  file: File;
  model_name: "resnet" | "vit";
}

export interface PredictResponse {
  model: string;
  prediction: number; // 0 | 1
  probability: number; // 0.0 – 1.0
}
