import { useForm } from "react-hook-form";
import type { FieldValues, SubmitHandler } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useEffect } from "react";
import type { ReactNode } from "react";
import type { ZodSchema } from "zod";
import ConfigLoader from "../configLoader/ConfigLoader";
import { useLocalConfig } from "../../hooks/useConfig";
import "./form.css";

interface SettingsFormProps<T extends FieldValues> {
   schema: ZodSchema<T>;
   onSubmit: (data: T) => void;
   settingsButton: {
      isOpen: boolean;
      handleButton: () => void;
   };
   className: string;
   submitLabel: string;
   children: (register: any, errors: any) => ReactNode;
   transformConfigForForm?: (config: any) => any;
   transformConfigForReset?: (config: any) => any;
}

function SettingsForm<T extends FieldValues>({
   schema,
   onSubmit,
   settingsButton,
   className,
   submitLabel,
   children,
   transformConfigForForm,
   transformConfigForReset,
}: SettingsFormProps<T>) {
   const {
      register,
      handleSubmit,
      formState: { errors },
      reset,
   } = useForm<T>({
      resolver: zodResolver(schema as any) as any,
   });

   const { config: loadedConfig } = useLocalConfig();

   useEffect(() => {
      if (loadedConfig) {
         const formValues = transformConfigForForm
            ? transformConfigForForm(loadedConfig)
            : loadedConfig;
         reset(formValues as T);
      }
   }, [loadedConfig, reset, transformConfigForForm]);

   const handleConfigLoad = (cfg: any) => {
      if (!cfg) return;

      const formValues = transformConfigForReset ? transformConfigForReset(cfg) : cfg;
      reset(formValues as T);
   };

   const onFormSubmit: SubmitHandler<T> = (data) => {
      onSubmit(data);
   };

   return (
      <aside className={`${className} ${settingsButton.isOpen ? `${className}--open` : ""}`}>
         <form className={`settings-form`} onSubmit={handleSubmit(onFormSubmit as any)}>
            {children(register, errors)}

            <div className="form__buttons">
               <button type="submit">{submitLabel}</button>
               <ConfigLoader onConfigLoad={handleConfigLoad} />
            </div>
         </form>
      </aside>
   );
}

export default SettingsForm;
