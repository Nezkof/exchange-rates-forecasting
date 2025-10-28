import type { ReactNode } from "react";
import type { FieldError } from "react-hook-form";

interface FormFieldProps {
   label: string;
   error?: FieldError;
   children: ReactNode;
}

export const FormField = ({ label, error, children }: FormFieldProps) => (
   <div className="form__item">
      <label>{label}</label>
      {children}
      {error && <span>{error.message}</span>}
   </div>
);

interface SelectFieldProps {
   label: string;
   register: any;
   name: string;
   options: { value: string; label: string }[];
   error?: FieldError;
}

export const SelectField = ({ label, register, name, options, error }: SelectFieldProps) => (
   <FormField label={label} error={error}>
      <select {...register(name)}>
         {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
               {opt.label}
            </option>
         ))}
      </select>
   </FormField>
);

interface InputFieldProps {
   label: string;
   register: any;
   name: string;
   type?: string;
   step?: string;
   placeholder?: string;
   error?: FieldError;
}

export const InputField = ({
   label,
   register,
   name,
   type = "text",
   step,
   placeholder,
   error,
}: InputFieldProps) => (
   <FormField label={label} error={error}>
      <input
         type={type}
         step={step}
         placeholder={placeholder}
         {...register(name, type === "number" ? { valueAsNumber: true } : {})}
      />
   </FormField>
);
