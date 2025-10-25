import "./optimizationSettings.css";
import "./form.css";

import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import ConfigLoader from "../configLoader/ConfigLoader";
import { useLocalConfig } from "../../hooks/useConfig";
import { useEffect } from "react";
import SettingsButton from "../settingsButton/SettingsButton";

const configSchema = z.object({
   csv_type: z.enum(["Returns", "Data"]),
   tickers: z
      .string()
      .min(1, "Tickers are required")
      .transform((val) => val.split(",").map((s) => s.trim().toUpperCase())),
   data_length: z.number().int().min(1),
   control_length: z.number().int().min(1),
   optimizer: z.enum(["ADAM", "SGD"]),
   window_size: z.number().int().min(1),
   hidden_size: z.number().int().min(1),
   samples_amount: z.number().int().min(1),
   risk_threshold: z.number().min(0, "Must be positive").max(100, "Cannot exceed 100"),
   capital: z.number().int().min(1),
});

type ConfigInput = z.input<typeof configSchema>;
type ConfigOutput = z.infer<typeof configSchema>;

interface Props {
   settingsButton: {
      isOpen: boolean;
      handleButton: () => void;
   };
   onSubmit: (data: ConfigOutput) => void;
}

const OptimizationSettings = ({ settingsButton, onSubmit }: Props) => {
   const {
      register,
      handleSubmit,
      formState: { errors },
      reset,
   } = useForm<ConfigInput>({
      resolver: zodResolver(configSchema),
   });

   const { config: loadedConfig } = useLocalConfig();

   useEffect(() => {
      if (loadedConfig) {
         const formValues: ConfigInput = {
            ...loadedConfig,
            tickers: loadedConfig.tickers.join(","),
            risk_threshold: loadedConfig.risk_threshold * 100,
         };
         reset(formValues);
      }
   }, [loadedConfig, reset]);

   return (
      <aside
         className={`optimization-settings ${
            settingsButton.isOpen ? "optimization-settings--open" : ""
         }`}
      >
         <SettingsButton isOpen={settingsButton.isOpen} handleBtn={settingsButton.handleButton} />

         <form className="optimization-settings__form" onSubmit={handleSubmit(onSubmit)}>
            {/* STRING FIELDS */}
            <div className="form__item">
               <label>CSV type</label>
               <select {...register("csv_type")}>
                  <option value="Returns">Returns</option>
                  <option value="Data">Data</option>
               </select>
               {errors.csv_type && <span>{errors.csv_type.message}</span>}
            </div>

            <div className="form__item">
               <label>Tickers</label>
               <input {...register("tickers")} placeholder="CNY,EUR,USD" />
               {errors.tickers && <span>{errors.tickers.message}</span>}
            </div>

            {/* NUMBER FIELDS */}
            <div className="form__item">
               <label>Hidden Size</label>
               <input
                  type="number"
                  {...register("hidden_size", { valueAsNumber: true })}
                  placeholder="256"
               />
               {errors.hidden_size && <span>{errors.hidden_size.message}</span>}
            </div>

            <div className="form__item">
               <label>Window Size</label>
               <input
                  type="number"
                  {...register("window_size", { valueAsNumber: true })}
                  placeholder="50"
               />
               {errors.window_size && <span>{errors.window_size.message}</span>}
            </div>

            <div className="form__item">
               <label>Optimizer</label>
               <select {...register("optimizer")}>
                  <option value="ADAM">ADAM</option>
                  <option value="SGD">SGD</option>
               </select>
               {errors.optimizer && <span>{errors.optimizer.message}</span>}
            </div>

            <div className="form__item">
               <label>Data Length</label>
               <input
                  type="number"
                  {...register("data_length", { valueAsNumber: true })}
                  placeholder="5000"
               />
               {errors.data_length && <span>{errors.data_length.message}</span>}
            </div>

            <div className="form__item">
               <label>Control Length</label>
               <input
                  type="number"
                  {...register("control_length", { valueAsNumber: true })}
                  placeholder="365"
               />
               {errors.control_length && <span>{errors.control_length.message}</span>}
            </div>

            <div className="form__item">
               <label>Samples Amount</label>
               <input
                  type="number"
                  {...register("samples_amount", { valueAsNumber: true })}
                  placeholder="5000"
               />
               {errors.samples_amount && <span>{errors.samples_amount.message}</span>}
            </div>

            <div className="form__item">
               <label>Risk Threshold (%)</label>
               <input
                  type="number"
                  {...register("risk_threshold", { valueAsNumber: true })}
                  placeholder="20%"
               />
               {errors.risk_threshold && <span>{errors.risk_threshold.message}</span>}
            </div>

            <div className="form__item">
               <label>Capital</label>
               <input
                  type="number"
                  {...register("capital", { valueAsNumber: true })}
                  placeholder="10000"
               />
               {errors.capital && <span>{errors.capital.message}</span>}
            </div>

            <div className="form__buttons">
               <button type="submit">Optimize</button>
               <ConfigLoader
                  onConfigLoad={(cfg) => {
                     const formValues: ConfigInput = {
                        ...cfg,
                        tickers: cfg.tickers.join(","),
                        risk_threshold: cfg.risk_threshold * 100,
                     };
                     reset(formValues);
                  }}
               />
            </div>
         </form>
      </aside>
   );
};

export default OptimizationSettings;
