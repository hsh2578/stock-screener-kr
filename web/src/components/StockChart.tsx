import { useEffect, useRef, useMemo } from 'react';
import {
  createChart,
  ColorType,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
} from 'lightweight-charts';
import type {
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  LineData,
  Time,
} from 'lightweight-charts';

export interface ChartDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface StockChartProps {
  data: ChartDataPoint[];
  height?: number;
  showMA?: boolean;
}

// 이동평균선 색상
const MA_COLORS = {
  ma5: '#FF69B4',   // 분홍
  ma20: '#4A90E2',  // 파랑
  ma40: '#E74C3C',  // 빨강
  ma150: '#95A5A6', // 회색
};

// 이동평균 계산
function calculateMA(data: ChartDataPoint[], period: number): LineData<Time>[] {
  const result: LineData<Time>[] = [];

  for (let i = period - 1; i < data.length; i++) {
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += data[i - j].close;
    }
    result.push({
      time: data[i].date as Time,
      value: sum / period,
    });
  }

  return result;
}

function StockChart({ data, height = 400, showMA = true }: StockChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  // 캔들스틱 데이터 변환
  const candlestickData: CandlestickData<Time>[] = useMemo(() => {
    return data.map((d) => ({
      time: d.date as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));
  }, [data]);

  // 거래량 데이터 변환
  const volumeData: HistogramData<Time>[] = useMemo(() => {
    return data.map((d, i) => ({
      time: d.date as Time,
      value: d.volume,
      color:
        i === 0
          ? 'rgba(74, 144, 226, 0.5)'
          : d.close >= data[i - 1].close
          ? 'rgba(255, 68, 68, 0.5)'
          : 'rgba(68, 68, 255, 0.5)',
    }));
  }, [data]);

  // 이동평균선 데이터
  const maData = useMemo(() => {
    if (!showMA || data.length < 5) return null;
    return {
      ma5: calculateMA(data, 5),
      ma20: data.length >= 20 ? calculateMA(data, 20) : [],
      ma40: data.length >= 40 ? calculateMA(data, 40) : [],
      ma150: data.length >= 150 ? calculateMA(data, 150) : [],
    };
  }, [data, showMA]);

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    // 차트 생성
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { type: ColorType.Solid, color: '#1A1A1A' },
        textColor: '#A0A0A0',
      },
      grid: {
        vertLines: { color: '#2A2A2A' },
        horzLines: { color: '#2A2A2A' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: '#4A90E2',
          width: 1,
          style: 2,
          labelBackgroundColor: '#4A90E2',
        },
        horzLine: {
          color: '#4A90E2',
          width: 1,
          style: 2,
          labelBackgroundColor: '#4A90E2',
        },
      },
      rightPriceScale: {
        borderColor: '#333333',
      },
      timeScale: {
        borderColor: '#333333',
        timeVisible: false,
      },
    });

    chartRef.current = chart;

    // 캔들스틱 시리즈 (v5 API)
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#FF4444',
      downColor: '#4444FF',
      borderUpColor: '#FF4444',
      borderDownColor: '#4444FF',
      wickUpColor: '#FF4444',
      wickDownColor: '#4444FF',
    });
    candlestickSeries.setData(candlestickData);
    candlestickSeriesRef.current = candlestickSeries;

    // 이동평균선
    if (maData) {
      if (maData.ma5.length > 0) {
        const ma5Series = chart.addSeries(LineSeries, {
          color: MA_COLORS.ma5,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        ma5Series.setData(maData.ma5);
      }

      if (maData.ma20.length > 0) {
        const ma20Series = chart.addSeries(LineSeries, {
          color: MA_COLORS.ma20,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        ma20Series.setData(maData.ma20);
      }

      if (maData.ma40.length > 0) {
        const ma40Series = chart.addSeries(LineSeries, {
          color: MA_COLORS.ma40,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        ma40Series.setData(maData.ma40);
      }

      if (maData.ma150.length > 0) {
        const ma150Series = chart.addSeries(LineSeries, {
          color: MA_COLORS.ma150,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        ma150Series.setData(maData.ma150);
      }
    }

    // 거래량 시리즈 (v5 API)
    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: 'rgba(74, 144, 226, 0.5)',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });
    volumeSeries.setData(volumeData);
    volumeSeriesRef.current = volumeSeries;

    // 차트 크기 조절
    chart.timeScale().fitContent();

    // 리사이즈 핸들러
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [candlestickData, volumeData, maData, height, data.length]);

  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height: height, backgroundColor: '#1A1A1A', borderRadius: '8px' }}
      >
        <p style={{ color: 'var(--color-text-muted)' }}>차트 데이터가 없습니다</p>
      </div>
    );
  }

  return (
    <div>
      <div ref={chartContainerRef} style={{ borderRadius: '8px', overflow: 'hidden' }} />

      {/* 범례 */}
      {showMA && (
        <div className="flex flex-wrap gap-4 mt-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
          <span style={{ color: MA_COLORS.ma5 }}>● 5일선</span>
          <span style={{ color: MA_COLORS.ma20 }}>● 20일선</span>
          <span style={{ color: MA_COLORS.ma40 }}>● 40일선</span>
          <span style={{ color: MA_COLORS.ma150 }}>● 150일선</span>
        </div>
      )}
    </div>
  );
}

export default StockChart;
