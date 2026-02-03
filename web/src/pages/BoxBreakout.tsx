import { useState } from 'react';
import {
  PageLayout,
  StockTable,
  ChartModal,
  LoadingState,
  ErrorState,
  formatNumber,
  formatPercent,
  getChangeColor,
} from '../components';
import type { Column } from '../components';
import { useScreenerData, useChartData } from '../hooks';

interface BoxBreakoutStock {
  ticker: string;
  name: string;
  price: number;
  change_rate: number;
  breakout_date: string;
  breakout_price: number;
  volume_ratio: number;
  ma150: number;
  updated_at: string;
}

function BoxBreakout() {
  const { data, meta, isLoading, error, refetch } = useScreenerData<BoxBreakoutStock>('box_breakout.json');
  const { getChartData } = useChartData();

  const [selectedStock, setSelectedStock] = useState<BoxBreakoutStock | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleRowClick = (stock: BoxBreakoutStock) => {
    setSelectedStock(stock);
    setIsModalOpen(true);
  };

  const columns: Column<BoxBreakoutStock>[] = [
    {
      key: 'name',
      header: '종목명',
      width: '180px',
      render: (_, row) => (
        <div>
          <div style={{ color: 'var(--color-text-primary)', fontWeight: 500 }}>
            {row.name}
          </div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {row.ticker}
          </div>
        </div>
      ),
    },
    {
      key: 'price',
      header: '현재가',
      align: 'right',
      render: (value) => formatNumber(value as number) + '원',
    },
    {
      key: 'change_rate',
      header: '등락률',
      align: 'right',
      render: (value) => (
        <span style={{ color: getChangeColor(value as number) }}>
          {formatPercent(value as number)}
        </span>
      ),
    },
    {
      key: 'breakout_date',
      header: '돌파일',
      align: 'center',
      render: (value) => {
        const date = new Date(value as string);
        return date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' });
      },
    },
    {
      key: 'breakout_price',
      header: '돌파가',
      align: 'right',
      render: (value) => formatNumber(value as number) + '원',
    },
    {
      key: 'volume_ratio',
      header: '거래량 배수',
      align: 'right',
      render: (value) => (
        <span style={{ color: (value as number) >= 2 ? 'var(--color-success)' : 'inherit' }}>
          {(value as number).toFixed(1)}배
        </span>
      ),
    },
    {
      key: 'ma150',
      header: '150일선',
      align: 'right',
      render: (value) => formatNumber(value as number) + '원',
    },
  ];

  return (
    <PageLayout
      icon="🚀"
      title="박스권 돌파 (거래량 동반)"
      description="박스권 돌파 + 거래량 2배 + 150일선 위"
      badge={{ text: '1차 매수', type: 'success' }}
      lastUpdated={meta?.lastUpdated}
      totalCount={meta?.totalCount}
      screenedFrom={meta?.screened_from}
    >
      {isLoading ? (
        <LoadingState type="table" rows={10} />
      ) : error ? (
        <ErrorState message={error} onRetry={refetch} />
      ) : (
        <div className="card overflow-hidden">
          <StockTable
            data={data}
            columns={columns}
            onRowClick={handleRowClick}
            keyField="ticker"
          />
        </div>
      )}

      {selectedStock && (
        <ChartModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          ticker={selectedStock.ticker}
          name={selectedStock.name}
          price={selectedStock.price}
          changeRate={selectedStock.change_rate}
          chartData={getChartData(selectedStock.ticker)}
        />
      )}
    </PageLayout>
  );
}

export default BoxBreakout;
