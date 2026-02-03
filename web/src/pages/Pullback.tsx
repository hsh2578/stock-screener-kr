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

interface PullbackStock {
  ticker: string;
  name: string;
  price: number;
  change_rate: number;
  breakout_date: string;
  resistance: number;
  pullback_ratio: number;
  volume_decrease: number;
  updated_at: string;
}

function Pullback() {
  const { data, meta, isLoading, error, refetch } = useScreenerData<PullbackStock>('pullback.json');
  const { getChartData } = useChartData();

  const [selectedStock, setSelectedStock] = useState<PullbackStock | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleRowClick = (stock: PullbackStock) => {
    setSelectedStock(stock);
    setIsModalOpen(true);
  };

  const columns: Column<PullbackStock>[] = [
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
      key: 'resistance',
      header: '저항선',
      align: 'right',
      render: (value) => formatNumber(value as number) + '원',
    },
    {
      key: 'pullback_ratio',
      header: '되돌림',
      align: 'right',
      render: (value) => {
        const ratio = Math.abs(value as number);
        return (
          <span style={{ color: ratio <= 2 ? 'var(--color-success)' : 'inherit' }}>
            {ratio.toFixed(1)}%
          </span>
        );
      },
    },
    {
      key: 'volume_decrease',
      header: '거래량 감소',
      align: 'right',
      render: (value) => (
        <span style={{ color: 'var(--color-success)' }}>
          -{(value as number).toFixed(0)}%
        </span>
      ),
    },
  ];

  return (
    <PageLayout
      icon="🎯"
      title="풀백 스크리너"
      description="돌파 후 저항선으로 되돌아온 종목"
      badge={{ text: '2차 매수', type: 'success' }}
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

export default Pullback;
