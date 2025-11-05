import { useEffect, useRef } from "react";
import { useLocalConfig } from "../../hooks/useConfig";

interface Props {
   onConfigLoad?: (cfg: any) => void;
}

const ConfigLoader = ({ onConfigLoad }: Props) => {
   const { config, error, handleFile } = useLocalConfig();
   const inputRef = useRef<HTMLInputElement>(null);

   useEffect(() => {
      if (config) {
         console.log(config);
         if (onConfigLoad) onConfigLoad(config);
      }
   }, [config]);

   useEffect(() => {
      if (error) console.error(error);
   }, [error]);

   const onLoadClick = () => {
      inputRef.current?.click();
   };

   const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files?.[0]) handleFile(e.target.files[0]);
   };

   return (
      <>
         <input
            type="file"
            accept=".json"
            style={{ display: "none" }}
            ref={inputRef}
            onChange={onFileChange}
         />
         <button type="button" onClick={onLoadClick}>
            Завантажити конфіг
         </button>
      </>
   );
};

export default ConfigLoader;
