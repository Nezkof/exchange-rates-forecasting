import type { Metrics } from "../../types/metrics";
import "./metricsTable.css";

interface Props {
   data: Metrics;
}

const metricDetails = {
   MAE: "Середня абсолютна похибка",
   MAPE: "Середня абсолютна процентна похибка",
   RMSE: "Середньоквадратична похибка",
};

const MetricsTable = ({ data }: Props) => {
   const metricKeys = Object.keys(data) as Array<keyof Metrics>;

   return (
      <table className="metrics-table">
         <thead>
            <tr>
               {metricKeys.map((key) => (
                  <th key={key}>
                     {key} ({metricDetails[key]})
                  </th>
               ))}
            </tr>
         </thead>
         <tbody>
            <tr>
               {metricKeys.map((key) => (
                  <td key={data[key]}>{data[key].toFixed(4)}</td>
               ))}
            </tr>
         </tbody>
      </table>
   );
};

export default MetricsTable;
