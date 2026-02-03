import { useState, useMemo } from 'react';

// 컬럼 정의 타입
export interface Column<T> {
  key: keyof T | string;
  header: string;
  width?: string;
  align?: 'left' | 'center' | 'right';
  sortable?: boolean;
  render?: (value: unknown, row: T) => React.ReactNode;
}

interface StockTableProps<T> {
  data: T[];
  columns: Column<T>[];
  onRowClick?: (row: T) => void;
  keyField?: keyof T;
}

type SortDirection = 'asc' | 'desc' | null;

// 숫자 포맷팅 유틸리티
export const formatNumber = (value: number): string => {
  return value.toLocaleString('ko-KR');
};

export const formatPercent = (value: number): string => {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
};

export const formatPrice = (value: number): string => {
  return value.toLocaleString('ko-KR') + '원';
};

// 등락률 색상 반환
export const getChangeColor = (value: number): string => {
  if (value > 0) return '#FF4444';
  if (value < 0) return '#4444FF';
  return 'var(--color-text-secondary)';
};

function StockTable<T extends object>({
  data,
  columns,
  onRowClick,
  keyField = 'ticker' as keyof T,
}: StockTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(null);

  // 정렬 처리
  const handleSort = (key: string) => {
    if (sortKey === key) {
      // 같은 컬럼 클릭: asc -> desc -> null
      if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else if (sortDirection === 'desc') {
        setSortDirection(null);
        setSortKey(null);
      } else {
        setSortDirection('asc');
      }
    } else {
      setSortKey(key);
      setSortDirection('asc');
    }
  };

  // 정렬된 데이터
  const sortedData = useMemo(() => {
    if (!sortKey || !sortDirection) return data;

    return [...data].sort((a, b) => {
      const aVal = a[sortKey as keyof T];
      const bVal = b[sortKey as keyof T];

      // null/undefined 처리
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;

      // 숫자 비교
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }

      // 문자열 비교
      const aStr = String(aVal);
      const bStr = String(bVal);
      return sortDirection === 'asc'
        ? aStr.localeCompare(bStr, 'ko')
        : bStr.localeCompare(aStr, 'ko');
    });
  }, [data, sortKey, sortDirection]);

  // 정렬 아이콘
  const getSortIcon = (key: string) => {
    if (sortKey !== key) return '⇅';
    if (sortDirection === 'asc') return '▲';
    if (sortDirection === 'desc') return '▼';
    return '⇅';
  };

  // 셀 값 가져오기
  const getCellValue = (row: T, key: string): unknown => {
    // nested key 지원 (예: "meta.name")
    const keys = key.split('.');
    let value: unknown = row;
    for (const k of keys) {
      if (value && typeof value === 'object') {
        value = (value as Record<string, unknown>)[k];
      } else {
        return undefined;
      }
    }
    return value;
  };

  if (data.length === 0) {
    return (
      <div className="card p-8 text-center">
        <p className="text-lg mb-2" style={{ color: 'var(--color-text-primary)' }}>
          조건에 맞는 종목이 없습니다
        </p>
        <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
          다른 스크리너를 확인해보세요.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                className={`px-4 py-3 text-sm font-medium ${
                  col.sortable !== false ? 'cursor-pointer select-none hover:opacity-80' : ''
                }`}
                style={{
                  color: 'var(--color-text-secondary)',
                  textAlign: col.align || 'left',
                  width: col.width,
                }}
                onClick={() => col.sortable !== false && handleSort(String(col.key))}
              >
                <span className="inline-flex items-center gap-1">
                  {col.header}
                  {col.sortable !== false && (
                    <span
                      className="text-xs opacity-50"
                      style={{
                        opacity: sortKey === String(col.key) ? 1 : 0.3,
                      }}
                    >
                      {getSortIcon(String(col.key))}
                    </span>
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row, rowIndex) => (
            <tr
              key={String(row[keyField]) || rowIndex}
              className="transition-colors"
              style={{
                borderBottom: '1px solid var(--color-border)',
                cursor: onRowClick ? 'pointer' : 'default',
              }}
              onClick={() => onRowClick?.(row)}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--color-bg-tertiary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              {columns.map((col) => {
                const value = getCellValue(row, String(col.key));
                return (
                  <td
                    key={String(col.key)}
                    className="px-4 py-3 text-sm"
                    style={{
                      color: 'var(--color-text-primary)',
                      textAlign: col.align || 'left',
                    }}
                  >
                    {col.render ? col.render(value, row) : String(value ?? '-')}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default StockTable;
