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
         <h2>Top 5 Portfolios (by Sharpe Ratio)</h2>
         <table className="portfolios-info__table">
            <thead>
               <tr className="portfolios-info__header-row">
                  <th>Risk</th>
                  <th>Return</th>
                  <th>Sharpe</th>
                  <th>Distribution</th>
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
                        <td data-label="Risk">{portfolio.risk.toFixed(3)}</td>
                        <td data-label="Return">{portfolio.return.toFixed(3)}</td>
                        <td data-label="Sharpe">{portfolio.sharpe_ratio.toFixed(3)}</td>
                        <td
                           data-label="Distribution (UAH)"
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
