import type { Portfolio } from "../../types/portfolio";
import "./portfoliosInfo.css";

interface Props {
   data: Portfolio[];
}

const PortfoliosInfo = ({ data }: Props) => {
   const sortedData = [...data].sort((a, b) => b.sharpe_ratio - a.sharpe_ratio);

   const top5Portfolios = sortedData.slice(0, 5);

   if (top5Portfolios.length === 0) {
      return (
         <aside className="portfolios-info">
            <h2>Top 5 Portfolios</h2>
            <p className="portfolios-info__empty">
               Portfolio data is not available yet. Run an optimization.
            </p>
         </aside>
      );
   }

   const formatCurrency = (amount: number) => {
      return amount.toLocaleString("en-US", {
         minimumFractionDigits: 2,
         maximumFractionDigits: 2,
      });
   };

   return (
      <aside className="portfolios-info">
         <h2>Топ 5 портфелів за коефіцієнтом Шарпа</h2>
         <table className="portfolios-info__table">
            <thead>
               <tr className="portfolios-info__header-row">
                  <th>Ризик</th>
                  <th>Прибуток</th>
                  <th>Коефіцієнт Шарпа</th>
                  <th>Розподіл</th>
               </tr>
            </thead>
            <tbody>
               {top5Portfolios.map((portfolio, index) => {
                  const distribution = portfolio.tickers
                     .map((ticker, i) => {
                        return `${ticker}: ${formatCurrency(portfolio.distribution[i])}`;
                     })
                     .join(", ");

                  return (
                     <tr key={index} className="portfolios-info__body-row">
                        <td data-label="Ризик">{portfolio.risk.toFixed(3)}</td>
                        <td data-label="Прибуток">{portfolio.return.toFixed(3)}</td>
                        <td data-label="Коефіцієнт Шарпа">{portfolio.sharpe_ratio.toFixed(3)}</td>
                        <td
                           data-label="Розподіл (UAH)"
                           className="portfolios-info__cell-distribution"
                        >
                           {distribution}
                        </td>
                     </tr>
                  );
               })}
            </tbody>
         </table>
      </aside>
   );
};

export default PortfoliosInfo;
