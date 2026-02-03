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

interface VolumeDryUpStock {
  ticker: string;
  name: string;
  price: number;
  explosion_date: string;
  explosion_change_rate: number;
  volume_decrease_rate: number;
  volume_rank: number;
  updated_at: string;
}

function VolumeDryUp() {
  const { data, meta, isLoading, error, refetch } = useScreenerData<VolumeDryUpStock>('volume_dry_up.json');
  const { getChartData } = useChartData();

  const [selectedStock, setSelectedStock] = useState<VolumeDryUpStock | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleRowClick = (stock: VolumeDryUpStock) => {
    setSelectedStock(stock);
    setIsModalOpen(true);
  };

  const columns: Column<VolumeDryUpStock>[] = [
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
      key: 'explosion_date',
      header: '폭발일',
      align: 'center',
      render: (value) => {
        const date = new Date(value as string);
        return date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' });
      },
    },
    {
      key: 'explosion_change_rate',
      header: '폭발일 상승',
      align: 'right',
      render: (value) => (
        <span style={{ color: getChangeColor(value as number) }}>
          {formatPercent(value as number)}
        </span>
      ),
    },
    {
      key: 'volume_decrease_rate',
      header: '거래량 감소',
      align: 'right',
      render: (value) => (
        <span style={{ color: 'var(--color-success)' }}>
          -{(value as number).toFixed(0)}%
        </span>
      ),
    },
    {
      key: 'volume_rank',
      header: '거래대금 순위',
      align: 'right',
      render: (value) => (
        <span style={{ color: (value as number) <= 30 ? 'var(--color-warning)' : 'inherit' }}>
          {value as number}위
        </span>
      ),
    },
  ];

  return (
    <PageLayout
      icon="🏜️"
      title="거래량 급감 눌림목"
      description="급등 후 매도세 고갈 종목"
      badge={{ text: '세력 보유', type: 'warning' }}
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
          changeRate={selectedStock.explosion_change_rate}
          chartData={getChartData(selectedStock.ticker)}
        />
      )}
    </PageLayout>
  );
}

export default VolumeDryUp;
