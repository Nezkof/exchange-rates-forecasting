import { useState } from "react";
import type { Config } from "../types/routes";

export const useLocalConfig = () => {
   const [config, setConfig] = useState<Config | null>(null);
   const [error, setError] = useState<string | null>(null);

   const handleFile = (file: File) => {
      const reader = new FileReader();
      reader.onload = (event) => {
         try {
            if (!event.target?.result) throw new Error("Empty file");
            const json = JSON.parse(event.target.result as string);
            setConfig(json);
            setError(null);
         } catch (err: any) {
            setError(err.message || "Failed to parse JSON");
         }
      };
      reader.readAsText(file);
   };

   return { config, error, handleFile };
};
