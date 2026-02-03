import { useState, useEffect, useCallback } from 'react';
import type { ChartDataPoint } from '../components/StockChart';

interface ChartDataStore {
  [ticker: string]: ChartDataPoint[];
}

interface UseChartDataReturn {
  chartData: ChartDataStore;
  getChartData: (ticker: string) => ChartDataPoint[];
  isLoading: boolean;
  error: string | null;
}

// 전역 차트 데이터 캐시
let globalChartData: ChartDataStore | null = null;
let globalError: string | null = null;
let loadPromise: Promise<void> | null = null;

export function useChartData(): UseChartDataReturn {
  const [chartData, setChartData] = useState<ChartDataStore>(globalChartData || {});
  const [isLoading, setIsLoading] = useState(!globalChartData);
  const [error, setError] = useState<string | null>(globalError);

  const fetchChartData = useCallback(async () => {
    // 이미 로딩 중이면 기존 Promise 사용
    if (loadPromise) {
      await loadPromise;
      setChartData(globalChartData || {});
      setIsLoading(false);
      setError(globalError);
      return;
    }

    // 이미 데이터가 있으면 스킵
    if (globalChartData) {
      setChartData(globalChartData);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);

    loadPromise = (async () => {
      try {
        const response = await fetch('/data/chart_data.json');

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data: ChartDataStore = await response.json();
        globalChartData = data;
        globalError = null;
      } catch (err) {
        console.error('Failed to fetch chart data:', err);
        globalError = '차트 데이터를 불러오지 못했습니다.';
        globalChartData = {};
      } finally {
        loadPromise = null;
      }
    })();

    await loadPromise;
    setChartData(globalChartData || {});
    setError(globalError);
    setIsLoading(false);
  }, []);

  useEffect(() => {
    fetchChartData();
  }, [fetchChartData]);

  const getChartData = useCallback(
    (ticker: string): ChartDataPoint[] => {
      return chartData[ticker] || [];
    },
    [chartData]
  );

  return {
    chartData,
    getChartData,
    isLoading,
    error,
  };
}

export default useChartData;
