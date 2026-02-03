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

interface VolumeExplosionStock {
  ticker: string;
  name: string;
  price: number;
  change_rate: number;
  volume_ratio: number;
  volume_rank: number;
  updated_at: string;
}

function VolumeExplosion() {
  const { data, meta, isLoading, error, refetch } = useScreenerData<VolumeExplosionStock>('volume_explosion.json');
  const { getChartData } = useChartData();

  const [selectedStock, setSelectedStock] = useState<VolumeExplosionStock | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleRowClick = (stock: VolumeExplosionStock) => {
    setSelectedStock(stock);
    setIsModalOpen(true);
  };

  const columns: Column<VolumeExplosionStock>[] = [
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
      key: 'volume_ratio',
      header: '거래량 배수',
      align: 'right',
      render: (value) => (
        <span style={{ color: (value as number) >= 10 ? 'var(--color-danger)' : 'var(--color-warning)' }}>
          {(value as number).toFixed(1)}배
        </span>
      ),
    },
    {
      key: 'volume_rank',
      header: '거래대금 순위',
      align: 'right',
      render: (value) => (
        <span style={{ color: (value as number) <= 10 ? 'var(--color-danger)' : (value as number) <= 30 ? 'var(--color-warning)' : 'inherit' }}>
          {value as number}위
        </span>
      ),
    },
  ];

  return (
    <PageLayout
      icon="💥"
      title="거래량 폭발"
      description="당일 거래량 6배 + 8% 이상 상승"
      badge={{ text: '당일', type: 'danger' }}
      lastUpdated={meta?.lastUpdated}
      totalCount={meta?.totalCount}
      screenedFrom={meta?.screened_from}
    >
      {/* 주의 문구 */}
      <div
        className="mb-6 p-4 rounded-lg text-sm"
        style={{
          backgroundColor: 'rgba(248, 113, 113, 0.1)',
          border: '1px solid rgba(248, 113, 113, 0.3)',
          color: 'var(--color-danger)',
        }}
      >
        ⚠️ 당일 급등 종목은 변동성이 매우 높습니다. 신중한 판단이 필요합니다.
      </div>

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

export default VolumeExplosion;
