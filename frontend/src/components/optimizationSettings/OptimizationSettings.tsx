import { z } from "zod";
import SettingsForm from "../settingsForm/SettingsForm";
import "./optimizationSettings.css";
import { InputField } from "../formField/FormField";

const configSchema = z.object({
   tickers: z
      .string()
      .min(1, "Tickers are required")
      .transform((val) => val.split(",").map((s) => s.trim().toUpperCase())),
   data_length: z.number().int().min(1),
   control_length: z.number().int().min(1),
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
   const transformForForm = (config: any): ConfigInput => ({
      ...config,
      tickers: config.tickers.join(","),
      risk_threshold: config.risk_threshold * 100,
   });

   return (
      <SettingsForm
         schema={configSchema}
         onSubmit={onSubmit}
         settingsButton={settingsButton}
         className="optimization-settings"
         submitLabel="Optimize"
         transformConfigForForm={transformForForm}
         transformConfigForReset={transformForForm}
      >
         {(register, errors) => (
            <>
               <InputField
                  label="Tickers"
                  register={register}
                  name="tickers"
                  placeholder="CNY,EUR,USD"
                  error={errors.tickers}
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

               <InputField
                  label="Samples Amount"
                  register={register}
                  name="samples_amount"
                  type="number"
                  placeholder="5000"
                  error={errors.samples_amount}
               />

               <InputField
                  label="Risk Threshold (%)"
                  register={register}
                  name="risk_threshold"
                  type="number"
                  placeholder="20"
                  error={errors.risk_threshold}
               />

               <InputField
                  label="Capital"
                  register={register}
                  name="capital"
                  type="number"
                  placeholder="10000"
                  error={errors.capital}
               />
            </>
         )}
      </SettingsForm>
   );
};

export default OptimizationSettings;
