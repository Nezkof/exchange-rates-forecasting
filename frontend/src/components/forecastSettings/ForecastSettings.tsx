import { z } from "zod";
import SettingsForm from "../settingsForm/SettingsForm";
import type { ForecastConfig } from "../../types/lstm";
import "./forecastSettings.css";
import { SelectField, InputField } from "../formField/FormField";
import { OPTIMIZER_OPTIONS } from "../../types/settings";

const configSchema = z.object({
   column_name: z.string(),
   data_length: z.number().int().min(1),
   control_length: z.number().int().min(1),
   optimizer: z.enum(["ADAM", "SGD"]),
   window_size: z.number().int().min(1),
   hidden_size: z.number().int().min(1),
});

type Config = z.infer<typeof configSchema>;

interface Props {
   settingsButton: {
      isOpen: boolean;
      handleButton: () => void;
   };
   onSubmit: (data: ForecastConfig) => void;
}

const ForecastSettings = ({ settingsButton, onSubmit }: Props) => {
   return (
      <SettingsForm
         schema={configSchema}
         onSubmit={onSubmit}
         settingsButton={settingsButton}
         className="train-settings"
         submitLabel="Forecast"
      >
         {(register, errors) => (
            <>
               <InputField
                  label="Column Name"
                  register={register}
                  name="column_name"
                  placeholder="USD"
                  error={errors.column_name}
               />

               <InputField
                  label="Hidden Size"
                  register={register}
                  name="hidden_size"
                  type="number"
                  placeholder="256"
                  error={errors.hidden_size}
               />

               <InputField
                  label="Window Size"
                  register={register}
                  name="window_size"
                  type="number"
                  placeholder="50"
                  error={errors.window_size}
               />

               <SelectField
                  label="Optimizer"
                  register={register}
                  name="optimizer"
                  options={OPTIMIZER_OPTIONS}
                  error={errors.optimizer}
               />

               <InputField
                  label="Data Length"
                  register={register}
                  name="data_length"
                  type="number"
                  placeholder="5000"
                  error={errors.data_length}
               />

               <InputField
                  label="Control Length"
                  register={register}
                  name="control_length"
                  type="number"
                  placeholder="365"
                  error={errors.control_length}
               />
            </>
         )}
      </SettingsForm>
   );
};

export default ForecastSettings;
