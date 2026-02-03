import { useEffect, useCallback } from 'react';
import StockChart from './StockChart';
import type { ChartDataPoint } from './StockChart';
import { formatNumber, formatPercent, getChangeColor } from './StockTable';

interface ChartModalProps {
  isOpen: boolean;
  onClose: () => void;
  ticker: string;
  name: string;
  price: number;
  changeRate?: number;
  chartData: ChartDataPoint[];
  isLoading?: boolean;
}

function ChartModal({
  isOpen,
  onClose,
  ticker,
  name,
  price,
  changeRate,
  chartData,
  isLoading = false,
}: ChartModalProps) {
  // ESC 키로 닫기
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    },
    [onClose]
  );

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backgroundColor: 'rgba(0, 0, 0, 0.8)' }}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      {/* 모달 컨테이너 */}
      <div
        className="w-full max-w-4xl max-h-[90vh] overflow-hidden rounded-xl animate-fadeIn"
        style={{
          backgroundColor: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 헤더 */}
        <div
          className="flex items-center justify-between p-4"
          style={{ borderBottom: '1px solid var(--color-border)' }}
        >
          <div className="flex items-center gap-4">
            {/* 종목 정보 */}
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-xl font-bold" style={{ color: 'var(--color-text-primary)' }}>
                  {name}
                </h2>
                <span className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                  {ticker}
                </span>
              </div>
              <div className="flex items-center gap-3 mt-1">
                <span className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                  {formatNumber(price)}원
                </span>
                {changeRate !== undefined && (
                  <span className="text-sm font-medium" style={{ color: getChangeColor(changeRate) }}>
                    {formatPercent(changeRate)}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* 닫기 버튼 */}
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:opacity-80 transition-opacity"
            style={{
              color: 'var(--color-text-secondary)',
              backgroundColor: 'var(--color-bg-tertiary)',
            }}
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* 차트 영역 */}
        <div className="p-4">
          {isLoading ? (
            <div
              className="flex items-center justify-center"
              style={{ height: 400, backgroundColor: '#1A1A1A', borderRadius: '8px' }}
            >
              <div className="text-center">
                <div
                  className="inline-block w-8 h-8 border-2 rounded-full animate-spin"
                  style={{
                    borderColor: 'var(--color-border)',
                    borderTopColor: 'var(--color-accent)',
                  }}
                />
                <p className="mt-2 text-sm" style={{ color: 'var(--color-text-muted)' }}>
                  차트 로딩 중...
                </p>
              </div>
            </div>
          ) : chartData.length === 0 ? (
            <div
              className="flex items-center justify-center"
              style={{ height: 400, backgroundColor: '#1A1A1A', borderRadius: '8px' }}
            >
              <p style={{ color: 'var(--color-text-muted)' }}>차트 데이터가 없습니다</p>
            </div>
          ) : (
            <StockChart data={chartData} height={400} showMA={true} />
          )}
        </div>

        {/* 푸터 */}
        <div
          className="flex items-center justify-between px-4 py-3 text-xs"
          style={{
            borderTop: '1px solid var(--color-border)',
            color: 'var(--color-text-muted)',
          }}
        >
          <span>데이터: KRX (한국거래소) · 수정주가 기준</span>
          <span>ESC 또는 바깥 영역 클릭으로 닫기</span>
        </div>
      </div>

      {/* 애니메이션 스타일 */}
      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: scale(0.95);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.2s ease-out;
        }
      `}</style>
    </div>
  );
}

export default ChartModal;
