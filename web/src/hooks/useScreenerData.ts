import { useState, useEffect, useCallback } from 'react';

interface ScreenerMeta {
  type: string;
  name: string;
  description: string;
  lastUpdated: string;
  totalCount: number;
  screened_from?: number;
}

interface ScreenerResponse<T> {
  meta: ScreenerMeta;
  data: T[];
}

interface UseScreenerDataReturn<T> {
  data: T[];
  meta: ScreenerMeta | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

// 메모리 캐시
const cache = new Map<string, { data: unknown; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5분

export function useScreenerData<T>(
  filename: string
): UseScreenerDataReturn<T> {
  const [data, setData] = useState<T[]>([]);
  const [meta, setMeta] = useState<ScreenerMeta | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // 캐시 확인
      const cached = cache.get(filename);
      if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        const response = cached.data as ScreenerResponse<T>;
        setData(response.data);
        setMeta(response.meta);
        setIsLoading(false);
        return;
      }

      const response = await fetch(`/data/${filename}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const json: ScreenerResponse<T> = await response.json();

      // 캐시 저장
      cache.set(filename, { data: json, timestamp: Date.now() });

      setData(json.data || []);
      setMeta(json.meta || null);
    } catch (err) {
      console.error('Failed to fetch screener data:', err);
      setError('데이터를 불러오지 못했습니다. 새로고침해주세요.');
      setData([]);
      setMeta(null);
    } finally {
      setIsLoading(false);
    }
  }, [filename]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    data,
    meta,
    isLoading,
    error,
    refetch: fetchData,
  };
}

export default useScreenerData;
