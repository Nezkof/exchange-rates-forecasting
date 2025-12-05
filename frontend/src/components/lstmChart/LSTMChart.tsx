import { Line } from "react-chartjs-2";
import type { LSTMTrainingResponse } from "../../types/lstm";
import { useRef } from "react";
import {
   Chart as ChartJS,
   type ChartOptions,
} from "chart.js";

import "./lstmChart.css";

interface LSTMChartProps {
   data: LSTMTrainingResponse;
}

export function LSTMChart({ data }: LSTMChartProps) {
   const chartRef = useRef<ChartJS<"line"> | null>(null);

   const chartData = {
      labels: [...data.train.dates, ...data.control.dates],
      datasets: [
         {
            label: "Результати тренування",
            data: [...data.train.results, ...Array(data.control.results.length).fill(null)],
            borderColor: "blue",
            backgroundColor: "rgba(0, 0, 255, 0.1)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1,
         },
         {
            label: "Очікувані результати тренування",
            data: [...data.train.expected, ...Array(data.control.expected.length).fill(null)],
            borderColor: "red",
            backgroundColor: "rgba(255, 0, 0, 0.1)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1,
         },
         {
            label: "Контрольні результати",
            data: [...Array(data.train.results.length).fill(null), ...data.control.results],
            borderColor: "lightblue",
            backgroundColor: "rgba(173, 216, 230, 0.1)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1,
         },
         {
            label: "Очікувані контрольні результати",
            data: [...Array(data.train.expected.length).fill(null), ...data.control.expected],
            borderColor: "pink",
            backgroundColor: "rgba(255, 192, 203, 0.1)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1,
         },
         // {
         //    label: "Pure Control Results",
         //    data: [...Array(data.train.results.length).fill(null), ...data.control.pure],
         //    borderColor: "yellow",
         //    backgroundColor: "rgba(255, 255, 0, 0.1)",
         //    borderWidth: 2,
         //    pointRadius: 0,
         //    tension: 0.1,
         // },
      ],
   };

   const options: ChartOptions<"line"> = {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
         mode: "index" as const,
         intersect: false,
      },
      plugins: {
         legend: {
            position: "top" as const,
            labels: {
               font: {
                  size: 14,
               },
               padding: 15,
            },
         },
         title: {
            display: true,
            text: "Результати LSTM",
            color: "№00000",
            font: {
               size: 18,
            },
         },
         zoom: {
            pan: {
               enabled: true,
               mode: "x" as const,
               modifierKey: "ctrl" as const,
            },
            zoom: {
               wheel: {
                  enabled: true,
                  speed: 0.1,
               },
               pinch: {
                  enabled: true,
               },
               mode: "x" as const,
            },
            limits: {
               x: { min: "original" as const, max: "original" as const },
               y: { min: "original" as const, max: "original" as const },
            },
         },
      },
      scales: {
         x: {
            display: true,
            title: {
               display: true,
               text: "Дата",
               font: {
                  size: 14,
               },
            },
            ticks: {
               maxRotation: 45,
               minRotation: 45,
               autoSkip: true,
               maxTicksLimit: 20,
               font: {
                  size: 10,
               },
            },
         },
         y: {
            display: true,
            title: {
               display: true,
               text: "Значення",
               font: {
                  size: 14,
               },
            },
            ticks: {
               font: {
                  size: 12,
               },
            },
         },
      },
   };

   const handleResetZoom = () => {
      if (chartRef.current) {
         chartRef.current.resetZoom();
      }
   };

   return (
      <div className="chart-wrapper">
         <div className="chart-controls">
            <button onClick={handleResetZoom} className="reset-zoom-btn">
               Скинути масштабування
            </button>
         </div>
         <div className="chart-container">
            <Line ref={chartRef} data={chartData} options={options} />
         </div>
      </div>
   );
}
