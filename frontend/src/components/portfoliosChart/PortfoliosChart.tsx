import "./portfoliosChart.css";
import { Scatter } from "react-chartjs-2";
import { useRef } from "react";
import type { Chart as ChartJS, ChartOptions, ChartData } from "chart.js";
import { Chart, LinearScale, PointElement, Tooltip, Legend, Title } from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import type { OptimizationResponse, Portfolio } from "../../types/portfolio";

Chart.register(LinearScale, PointElement, Tooltip, Legend, Title, zoomPlugin);

interface PortfoliosChartProps {
   data: OptimizationResponse;
}

const PortfoliosChart = ({ data }: PortfoliosChartProps) => {
   const chartRef = useRef<ChartJS<"scatter"> | null>(null);
   const scatterData = data.portfolios.map((p: Portfolio) => ({
      x: p.risk,
      y: p.return,
      sharpe: p.sharpe_ratio,
      tickers: p.tickers.join(", "),
      weights: p.distribution.map((w) => w.toFixed(2)).join(", "),
   }));

   const chartData: ChartData<"scatter"> = {
      datasets: [
         {
            label: "Portfolios",
            data: scatterData,
            backgroundColor: "rgba(0, 0, 255, 0.5)",
            borderColor: "rgba(0, 0, 255, 1)",
            borderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 7,
         },
      ],
   };

   const options: ChartOptions<"scatter"> = {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
         mode: "nearest" as const,
         intersect: false,
      },
      plugins: {
         legend: {
            position: "top" as const,
         },
         title: {
            display: true,
            text: data.metadata.chart_title,
            font: {
               size: 18,
            },
         },
         tooltip: {
            callbacks: {
               label: function (context) {
                  const point = context.raw as {
                     x: number;
                     y: number;
                     sharpe: number;
                     tickers: string;
                     weights: string;
                  };
                  if (!point) return "";

                  return [
                     `Risk: ${point.x.toFixed(3)}`,
                     `Return: ${point.y.toFixed(3)}`,
                     `Sharpe: ${point.sharpe.toFixed(3)}`,
                     `---`,
                     `Tickers: ${point.tickers}`,
                     `Weights: ${point.weights}`,
                  ];
               },
            },
         },
         zoom: {
            pan: {
               enabled: true,
               mode: "xy" as const,
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
               mode: "xy" as const,
            },
            limits: {
               x: { min: data.metadata.x_range[0], max: data.metadata.x_range[1] },
               y: { min: data.metadata.y_range[0], max: data.metadata.y_range[1] },
            },
         },
      },
      scales: {
         x: {
            type: "linear" as const,
            display: true,
            title: {
               display: true,
               text: data.metadata.x_axis_label,
               font: { size: 14 },
            },
            min: data.metadata.x_range[0],
            max: data.metadata.x_range[1],
         },
         y: {
            type: "linear" as const,
            display: true,
            title: {
               display: true,
               text: data.metadata.y_axis_label,
               font: { size: 14 },
            },
            min: data.metadata.y_range[0],
            max: data.metadata.y_range[1],
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
               Reset Zoom
            </button>
            <span className="chart-hint">Scroll to zoom | Ctrl + Drag to pan</span>
         </div>
         <div className="chart-container">
            <Scatter ref={chartRef} data={chartData} options={options} />
         </div>
      </div>
   );
};

export default PortfoliosChart;
