import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

import "./trainSettings.css";
import "./form.css";

import ConfigLoader from "../configLoader/ConfigLoader";
import { useLocalConfig } from "../../hooks/useConfig";
import { useEffect } from "react";
import type { SettingsConfig } from "../../types/lstm";
import SettingsButton from "../settingsButton/SettingsButton";

const configSchema = z.object({
   csv_type: z.enum(["Returns", "Data"]),
   column_name: z.string(),
   hidden_size: z.number().int().min(1),
   window_size: z.number().int().min(1),
   batch_size: z.number().int().min(1),
   learning_rate: z.number().positive(),
   learning_rate_decrease_speed: z.number().positive(),
   epochs: z.number().int().min(1),
   precision: z.number().positive(),
   optimizer: z.enum(["ADAM", "SGD"]),

   data_length: z.number().int().min(1),
   control_length: z.number().int().min(1),
});

type Config = z.infer<typeof configSchema>;

interface Props {
   settingsButton: {
      isSettingsOpen: boolean;
      handleButton: () => void;
   };
   onSubmit: (data: SettingsConfig) => void;
}

const TrainSettings = ({ settingsButton, onSubmit }: Props) => {
   const {
      register,
      handleSubmit,
      formState: { errors },
      reset,
   } = useForm<Config>({
      resolver: zodResolver(configSchema),
   });

   const { config: loadedConfig } = useLocalConfig();

   useEffect(() => {
      if (loadedConfig) {
         reset(loadedConfig);
      }
   }, [loadedConfig, reset]);

   return (
      <aside
         className={`train-settings ${settingsButton.isSettingsOpen ? "train-settings--open" : ""}`}
      >
         <SettingsButton
            isOpen={settingsButton.isSettingsOpen}
            handleBtn={settingsButton.handleButton}
         />

         <form className="train-settings__form" onSubmit={handleSubmit(onSubmit)}>
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
               <label>Column Name</label>
               <input {...register("column_name")} />
               {errors.column_name && <span>{errors.column_name.message}</span>}
            </div>

            {/* NUMBER FIELDS */}
            <div className="form__item">
               <label>Hidden Size</label>
               <input type="number" {...register("hidden_size", { valueAsNumber: true })} />
               {errors.hidden_size && <span>{errors.hidden_size.message}</span>}
            </div>

            <div className="form__item">
               <label>Window Size</label>
               <input type="number" {...register("window_size", { valueAsNumber: true })} />
               {errors.window_size && <span>{errors.window_size.message}</span>}
            </div>

            <div className="form__item">
               <label>Batch Size</label>
               <input type="number" {...register("batch_size", { valueAsNumber: true })} />
               {errors.batch_size && <span>{errors.batch_size.message}</span>}
            </div>

            <div className="form__item">
               <label>Learning Rate</label>
               <input
                  type="number"
                  step="any"
                  {...register("learning_rate", { valueAsNumber: true })}
               />
               {errors.learning_rate && <span>{errors.learning_rate.message}</span>}
            </div>

            <div className="form__item">
               <label>Learning Rate Decrease Speed</label>
               <input
                  type="number"
                  step="any"
                  {...register("learning_rate_decrease_speed", { valueAsNumber: true })}
               />
               {errors.learning_rate_decrease_speed && (
                  <span>{errors.learning_rate_decrease_speed.message}</span>
               )}
            </div>

            <div className="form__item">
               <label>Epochs</label>
               <input type="number" {...register("epochs", { valueAsNumber: true })} />
               {errors.epochs && <span>{errors.epochs.message}</span>}
            </div>

            <div className="form__item">
               <label>Precision</label>
               <input
                  type="number"
                  step="any"
                  {...register("precision", { valueAsNumber: true })}
               />
               {errors.precision && <span>{errors.precision.message}</span>}
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
               <input type="number" {...register("data_length", { valueAsNumber: true })} />
               {errors.data_length && <span>{errors.data_length.message}</span>}
            </div>

            <div className="form__item">
               <label>Control Length</label>
               <input type="number" {...register("control_length", { valueAsNumber: true })} />
               {errors.control_length && <span>{errors.control_length.message}</span>}
            </div>

            <div className="form__buttons">
               <button type="submit">Train</button>
               <ConfigLoader onConfigLoad={(cfg) => reset(cfg)} />
            </div>
         </form>
      </aside>
   );
};

export default TrainSettings;
