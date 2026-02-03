import {
  PageLayout,
  LoadingState,
  ErrorState,
  formatNumber,
  formatPercent,
} from '../components';
import { useScreenerData } from '../hooks';

interface SectorStageData {
  sector_name: string;
  stage: number;
  stage_name: string;
  ma150_slope: number;
  return_3m: number;
  is_overheated: boolean;
  current_price: number;
  ma150: number;
  updated_at: string;
}

// 단계별 스타일 정의
const STAGE_STYLES: Record<number, { color: string; bgColor: string; borderColor: string; emoji: string }> = {
  1: {
    color: 'var(--color-text-secondary)',
    bgColor: 'rgba(156, 163, 175, 0.1)',
    borderColor: 'rgba(156, 163, 175, 0.3)',
    emoji: '1️⃣',
  },
  2: {
    color: 'var(--color-success)',
    bgColor: 'rgba(74, 222, 128, 0.1)',
    borderColor: 'rgba(74, 222, 128, 0.3)',
    emoji: '2️⃣',
  },
  3: {
    color: 'var(--color-warning)',
    bgColor: 'rgba(251, 191, 36, 0.1)',
    borderColor: 'rgba(251, 191, 36, 0.3)',
    emoji: '3️⃣',
  },
  4: {
    color: 'var(--color-danger)',
    bgColor: 'rgba(248, 113, 113, 0.1)',
    borderColor: 'rgba(248, 113, 113, 0.3)',
    emoji: '4️⃣',
  },
};

function SectorCard({ sector }: { sector: SectorStageData }) {
  const style = STAGE_STYLES[sector.stage] || STAGE_STYLES[1];
  const priceVsMa = ((sector.current_price - sector.ma150) / sector.ma150) * 100;

  return (
    <div
      className="card p-4 transition-all duration-200 hover:scale-[1.02]"
      style={{
        backgroundColor: style.bgColor,
        border: `1px solid ${style.borderColor}`,
      }}
    >
      {/* 헤더: 업종명 + 단계 */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold truncate" style={{ color: 'var(--color-text-primary)' }}>
          {sector.sector_name}
        </h3>
        <div className="flex items-center gap-1">
          <span className="text-lg">{style.emoji}</span>
          <span className="text-sm font-medium" style={{ color: style.color }}>
            {sector.stage_name}
          </span>
        </div>
      </div>

      {/* 지표들 */}
      <div className="space-y-2 text-sm">
        {/* 현재 지수 */}
        <div className="flex justify-between">
          <span style={{ color: 'var(--color-text-muted)' }}>현재 지수</span>
          <span style={{ color: 'var(--color-text-primary)' }}>
            {formatNumber(sector.current_price)}
          </span>
        </div>

        {/* 150일선 대비 */}
        <div className="flex justify-between">
          <span style={{ color: 'var(--color-text-muted)' }}>150일선 대비</span>
          <span style={{ color: priceVsMa >= 0 ? 'var(--color-success)' : 'var(--color-danger)' }}>
            {formatPercent(priceVsMa)}
          </span>
        </div>

        {/* 150일선 기울기 */}
        <div className="flex justify-between">
          <span style={{ color: 'var(--color-text-muted)' }}>MA 기울기</span>
          <span style={{ color: sector.ma150_slope >= 0 ? 'var(--color-success)' : 'var(--color-danger)' }}>
            {formatPercent(sector.ma150_slope)}
          </span>
        </div>

        {/* 3개월 수익률 */}
        <div className="flex justify-between">
          <span style={{ color: 'var(--color-text-muted)' }}>3개월 수익률</span>
          <span style={{ color: sector.return_3m >= 0 ? 'var(--color-success)' : 'var(--color-danger)' }}>
            {formatPercent(sector.return_3m)}
          </span>
        </div>
      </div>

      {/* 과열 경고 */}
      {sector.is_overheated && (
        <div
          className="mt-3 px-2 py-1 rounded text-xs text-center"
          style={{
            backgroundColor: 'rgba(248, 113, 113, 0.2)',
            color: 'var(--color-danger)',
          }}
        >
          🔥 6개월 연속 상승 - 과열 주의
        </div>
      )}
    </div>
  );
}

function SectorStage() {
  const { data, meta, isLoading, error, refetch } = useScreenerData<SectorStageData>('sector_stage.json');

  // 단계별 그룹핑
  const groupedByStage = data.reduce((acc, sector) => {
    const stage = sector.stage;
    if (!acc[stage]) acc[stage] = [];
    acc[stage].push(sector);
    return acc;
  }, {} as Record<number, SectorStageData[]>);

  return (
    <PageLayout
      icon="🏭"
      title="업종별 4단계"
      description="Weinstein 4단계 업종 분석"
      lastUpdated={meta?.lastUpdated}
      totalCount={meta?.totalCount}
    >
      {/* 단계 설명 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="card p-4 text-center">
          <div className="text-2xl mb-2">1️⃣</div>
          <div className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>기초</div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>횡보 구간</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-2xl mb-2">2️⃣</div>
          <div className="font-semibold" style={{ color: 'var(--color-success)' }}>상승</div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>매수 적기</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-2xl mb-2">3️⃣</div>
          <div className="font-semibold" style={{ color: 'var(--color-warning)' }}>최정상</div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>매도 준비</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-2xl mb-2">4️⃣</div>
          <div className="font-semibold" style={{ color: 'var(--color-danger)' }}>쇠퇴</div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>매도/관망</div>
        </div>
      </div>

      {isLoading ? (
        <LoadingState type="card" rows={12} />
      ) : error ? (
        <ErrorState message={error} onRetry={refetch} />
      ) : (
        <div className="space-y-8">
          {/* 2단계 (상승) - 가장 중요 */}
          {groupedByStage[2]?.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span style={{ color: 'var(--color-success)' }}>2️⃣ 상승 단계</span>
                <span
                  className="text-sm px-2 py-0.5 rounded"
                  style={{ backgroundColor: 'rgba(74, 222, 128, 0.2)', color: 'var(--color-success)' }}
                >
                  매수 적기
                </span>
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {groupedByStage[2].map((sector) => (
                  <SectorCard key={sector.sector_name} sector={sector} />
                ))}
              </div>
            </section>
          )}

          {/* 1단계 (기초) */}
          {groupedByStage[1]?.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span style={{ color: 'var(--color-text-secondary)' }}>1️⃣ 기초 단계</span>
                <span
                  className="text-sm px-2 py-0.5 rounded"
                  style={{ backgroundColor: 'rgba(156, 163, 175, 0.2)', color: 'var(--color-text-secondary)' }}
                >
                  횡보 구간
                </span>
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {groupedByStage[1].map((sector) => (
                  <SectorCard key={sector.sector_name} sector={sector} />
                ))}
              </div>
            </section>
          )}

          {/* 3단계 (최정상) */}
          {groupedByStage[3]?.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span style={{ color: 'var(--color-warning)' }}>3️⃣ 최정상 단계</span>
                <span
                  className="text-sm px-2 py-0.5 rounded"
                  style={{ backgroundColor: 'rgba(251, 191, 36, 0.2)', color: 'var(--color-warning)' }}
                >
                  매도 준비
                </span>
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {groupedByStage[3].map((sector) => (
                  <SectorCard key={sector.sector_name} sector={sector} />
                ))}
              </div>
            </section>
          )}

          {/* 4단계 (쇠퇴) */}
          {groupedByStage[4]?.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span style={{ color: 'var(--color-danger)' }}>4️⃣ 쇠퇴 단계</span>
                <span
                  className="text-sm px-2 py-0.5 rounded"
                  style={{ backgroundColor: 'rgba(248, 113, 113, 0.2)', color: 'var(--color-danger)' }}
                >
                  매도/관망
                </span>
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {groupedByStage[4].map((sector) => (
                  <SectorCard key={sector.sector_name} sector={sector} />
                ))}
              </div>
            </section>
          )}

          {/* 데이터 없음 */}
          {data.length === 0 && (
            <div className="card p-8 text-center">
              <p className="text-lg mb-2" style={{ color: 'var(--color-text-primary)' }}>
                분석 결과가 없습니다
              </p>
              <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                데이터가 업데이트되면 결과가 표시됩니다.
              </p>
            </div>
          )}
        </div>
      )}
    </PageLayout>
  );
}

export default SectorStage;
