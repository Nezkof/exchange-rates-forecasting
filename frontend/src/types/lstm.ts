import type { Metrics } from "./metrics";

export interface TrainingDataset {
   dates: string[];
   results: number[];
   expected: number[];
}

export interface ControlDataset extends TrainingDataset {
   pure: number[];
}

export interface LSTMTrainingResponse {
   train: TrainingDataset;
   control: ControlDataset;
}

export interface LSTMForecastResponse {
   train: TrainingDataset;
   control: ControlDataset;
   metrics: Metrics;
}

export type Optimizer = "ADAM" | "SGD";
export type CSV_TYPE = "Returns" | "Data";

export interface TrainConfig {
   column_name: string;
   hidden_size: number;
   window_size: number;
   batch_size: number;
   learning_rate: number;
   learning_rate_decrease_speed: number;
   epochs: number;
   precision: number;
   optimizer: Optimizer;
   data_length: number;
   control_length: number;
}

export interface ForecastConfig {
   column_name: string;
   data_length: number;
   control_length: number;
   optimizer: Optimizer;
   window_size: number;
   hidden_size: number;
}
