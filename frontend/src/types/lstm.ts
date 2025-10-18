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

export type Optimizer = "ADAM" | "SGD";
export type CSV_TYPE = "Returns" | "Data";

export interface SettingsConfig {
   csv_type: CSV_TYPE;
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
