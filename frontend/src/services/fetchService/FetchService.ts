import axios from "axios";
import type { Config } from "../../types/routes";

class FetchService {
   static readonly BASE_URL = "http://127.0.0.1:8000/api/v1";

   static async trainModel(data: Config) {
      try {
         const response = await axios.post(`${this.BASE_URL}/lstm/train`, data);
         console.log("Response:", response.data);
         return response.data;
      } catch (error) {
         console.error("Error training model:", error);
         throw error;
      }
   }
}

export default FetchService;
