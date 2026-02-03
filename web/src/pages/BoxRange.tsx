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

interface BoxRangeStock {
  ticker: string;
  name: string;
  price: number;
  change_rate: number;
  box_high: number;
  box_low: number;
  range_percent: number;       // 변동폭 (%)
  atr: number;                 // ATR 값 (%)
  atr_multiple: number;        // 변동폭 / ATR
  support_touches: number;     // 지지선 터치 횟수
  resistance_touches: number;  // 저항선 터치 횟수
  slope_daily: number;         // 일평균 기울기 (%)
  volume_decrease_rate: number; // 거래량 감소율 (%)
  days: number;                // 횡보 기간 (일)
  updated_at: string;
}

function BoxRange() {
  const { data, meta, isLoading, error, refetch } = useScreenerData<BoxRangeStock>('box_range.json');
  const { getChartData } = useChartData();

  const [selectedStock, setSelectedStock] = useState<BoxRangeStock | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleRowClick = (stock: BoxRangeStock) => {
    setSelectedStock(stock);
    setIsModalOpen(true);
  };

  const columns: Column<BoxRangeStock>[] = [
    {
      key: 'name',
      header: '종목명',
      width: '160px',
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
      key: 'box_high',
      header: '박스 상단',
      align: 'right',
      render: (value) => formatNumber(value as number) + '원',
    },
    {
      key: 'box_low',
      header: '박스 하단',
      align: 'right',
      render: (value) => formatNumber(value as number) + '원',
    },
    {
      key: 'range_percent',
      header: '변동폭',
      align: 'right',
      render: (value) => formatPercent(value as number),
    },
    {
      key: 'support_touches',
      header: '지지 터치',
      align: 'center',
      width: '80px',
      render: (_, row) => (
        <span title={`지지선 ${row.support_touches}회 / 저항선 ${row.resistance_touches}회`}>
          {row.support_touches}/{row.resistance_touches}
        </span>
      ),
    },
    {
      key: 'atr_multiple',
      header: 'ATR배수',
      align: 'right',
      width: '80px',
      render: (value) => `${(value as number).toFixed(1)}x`,
    },
    {
      key: 'days',
      header: '횡보 기간',
      align: 'right',
      width: '80px',
      render: (value) => `${value}일`,
    },
  ];

  return (
    <PageLayout
      icon="📦"
      title="박스권 스크리너"
      description="60거래일 ATR기반 횡보 + 지지/저항 터치 확인 + 추세 필터 + 거래량 감소 (7조건)"
      badge={{ text: '퀀트', type: 'info' }}
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

      {/* 차트 모달 */}
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

export default BoxRange;
