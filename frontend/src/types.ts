export interface PredictRequest {
  file: File;
  model_name: "resnet" | "vit";
}

export interface PredictResponse {
  model: string;
  prediction: number; // 0 | 1
  probability_fake: number; // 0.0 – 1.0
  threshold: number; // 0.0 – 1.0
}
