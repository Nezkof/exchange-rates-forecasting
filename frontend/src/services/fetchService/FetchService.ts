import axios from "axios";
import type { LSTMTrainingResponse, SettingsConfig } from "../../types/lstm";

class FetchService {
   private readonly BASE_URL = "http://127.0.0.1:8000/api/v1";

   async trainModel(config: SettingsConfig): Promise<LSTMTrainingResponse> {
      const response = await axios.post(`${this.BASE_URL}/lstm/train`, config);

      return response.data;
   }
}

export default new FetchService();
