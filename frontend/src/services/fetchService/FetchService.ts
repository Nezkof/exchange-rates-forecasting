import axios from "axios";
import type {
   ForecastConfig,
   LSTMForecastResponse,
   LSTMTrainingResponse,
   TrainConfig,
} from "../../types/lstm";
import type { OptimizationConfig, OptimizationResponse } from "../../types/portfolio";

class FetchService {
   private readonly BASE_URL = "http://127.0.0.1:8000/api/v1";

   async train(config: TrainConfig): Promise<LSTMTrainingResponse> {
      const response = await axios.post(`${this.BASE_URL}/lstm/train`, config);
      return response.data;
   }

   async forecast(config: ForecastConfig): Promise<LSTMForecastResponse> {
      const response = await axios.post(`${this.BASE_URL}/lstm/forecast`, config);
      return response.data;
   }

   async optimize(config: OptimizationConfig): Promise<OptimizationResponse> {
      console.log(config);

      const response = await axios.post(`${this.BASE_URL}/portfolio/optimize`, config);
      return response.data;
   }
}

export default new FetchService();
