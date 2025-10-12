import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

import "./trainSettings.css";
import "./form.css";

import ChevronsLeft from "/icons/chevronsLeft.svg";

const configSchema = z.object({
   csv_path: z.string(),
   weights_path: z.string(),
   column_name: z.string(),

   hidden_size: z.number().int().min(1),
   output_size: z.number().int().min(1),
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

const keys = Object.keys(configSchema.shape);

const labelsMap: Record<string, string> = {
   csv_path: "CSV Path",
   weights_path: "Weights Path",
   column_name: "Column Name",
   hidden_size: "Hidden Size",
   output_size: "Output Size",
   window_size: "Window Size",
   batch_size: "Batch Size",
   learning_rate: "Learning Rate",
   learning_rate_decrease_speed: "Learning Rate Decrease Speed",
   epochs: "Epochs",
   precision: "Precision",
   optimizer: "Optimizer",
   data_length: "Data Length",
   control_length: "Control Length",
};

type Config = z.infer<typeof configSchema>;

interface Props {
   settingsButton: {
      isSettingsOpen: boolean;
      handleButton: () => void;
   };
}

const TrainSettings = ({ settingsButton }: Props) => {
   const {
      register,
      handleSubmit,
      formState: { errors },
   } = useForm<Config>({
      resolver: zodResolver(configSchema),
      // defaultValues: initialConfig,
   });

   const onSubmit = (data: Config) => {
      console.log("✅ Config saved:", data);
   };

   return (
      <aside
         className={`train-settings ${settingsButton.isSettingsOpen ? "train-settings--open" : ""}`}
      >
         <button
            className="train-settings__button train-settings__button--open"
            onClick={settingsButton.handleButton}
         >
            <img src={ChevronsLeft} alt="close" />
         </button>
         <h1 className="train-setting__header"></h1>
         <form className="train-settings__form" onSubmit={handleSubmit(onSubmit)}>
            {keys.map((key) => (
               <div className="form__item" key={key}>
                  <label>{labelsMap[key] || key}</label>
                  <input {...register(key as keyof Config)} />
                  {errors[key as keyof Config] && <p>{errors[key as keyof Config]?.message}</p>}
               </div>
            ))}

            <div className="form__buttons">
               <button type="button">Train</button>
               <button type="submit">Save config</button>
               <button
                  type="button"
                  // onClick={onLoad}
               >
                  Load config
               </button>
            </div>
         </form>
      </aside>
   );
};

export default TrainSettings;
