import { z } from "zod";
import SettingsForm from "../settingsForm/SettingsForm";
import type { TrainConfig } from "../../types/lstm";
import "./trainSettings.css";
import { InputField, SelectField } from "../formField/FormField";
import { OPTIMIZER_OPTIONS } from "../../types/settings";

const configSchema = z.object({
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

interface Props {
   settingsButton: {
      isOpen: boolean;
      handleButton: () => void;
   };
   onSubmit: (data: TrainConfig) => void;
}

const TrainSettings = ({ settingsButton, onSubmit }: Props) => {
   return (
      <SettingsForm
         schema={configSchema}
         onSubmit={onSubmit}
         settingsButton={settingsButton}
         className="train-settings"
         submitLabel="Train"
      >
         {(register, errors) => (
            <>
               <InputField
                  label="Назва колонки"
                  register={register}
                  name="column_name"
                  placeholder="USD"
                  error={errors.column_name}
               />
               <InputField
                  label="Розмір прихованого шару"
                  register={register}
                  name="hidden_size"
                  type="number"
                  placeholder="256"
                  error={errors.hidden_size}
               />
               <InputField
                  label="Розмір вікнА"
                  register={register}
                  name="window_size"
                  type="number"
                  placeholder="50"
                  error={errors.window_size}
               />
               <InputField
                  label="Розмір батча"
                  register={register}
                  name="batch_size"
                  type="number"
                  placeholder="64"
                  error={errors.batch_size}
               />
               <InputField
                  label="Коеф. швидкості навчання"
                  register={register}
                  name="learning_rate"
                  type="number"
                  step="any"
                  placeholder="0.001"
                  error={errors.learning_rate}
               />
               <InputField
                  label="Коеф. зменшення швидкості навчання"
                  register={register}
                  name="learning_rate_decrease_speed"
                  type="number"
                  step="any"
                  placeholder="0.0001"
                  error={errors.learning_rate_decrease_speed}
               />
               <InputField
                  label="Кількість епох"
                  register={register}
                  name="epochs"
                  type="number"
                  placeholder="3000"
                  error={errors.epochs}
               />
               <InputField
                  label="Точність"
                  register={register}
                  name="precision"
                  type="number"
                  step="any"
                  placeholder="0.0001"
                  error={errors.precision}
               />
               <SelectField
                  label="Оптимізатор"
                  register={register}
                  name="optimizer"
                  options={OPTIMIZER_OPTIONS}
                  error={errors.optimizer}
               />
               <InputField
                  label="Довжина даних"
                  register={register}
                  name="data_length"
                  type="number"
                  placeholder="5000"
                  error={errors.data_length}
               />
               <InputField
                  label="Довжина контрольної вибірки"
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

export default TrainSettings;
