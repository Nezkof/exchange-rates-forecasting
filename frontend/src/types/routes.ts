export const routes = [
   { to: "/", label: "Home" },
   { to: "/training", label: "Training" },
   { to: "/forecasting", label: "Forecasting" },
   { to: "/optimization", label: "Optimization" },
];

export type Optimizer = "ADAM" | "SGD";

export interface Config {
   csv_path: string;
   weights_path: string;
   column_name: string;
   hidden_size: number;
   output_size: number;
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
