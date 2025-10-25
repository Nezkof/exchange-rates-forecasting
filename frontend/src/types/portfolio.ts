export interface OptimizationConfig {
   csv_type: string;
   tickers: string[];
   data_length: number;
   control_length: number;
   optimizer: string;
   window_size: number;
   hidden_size: number;
   samples_amount: number;
   risk_threshold: number;
   capital: number;
}

export interface Portfolio {
   risk: number;
   return: number;
   sharpe_ratio: number;
   tickers: string[];
   distribution: number[];
}

export interface ChartMetadata {
   x_axis_label: string;
   y_axis_label: string;
   color_bar_label: string;
   chart_title: string;
   x_range: [number, number];
   y_range: [number, number];
}

export interface OptimizationResponse {
   portfolios: Portfolio[];
   metadata: ChartMetadata;
}
