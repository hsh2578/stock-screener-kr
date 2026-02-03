import { Link } from 'react-router-dom';
import { useEffect, useState } from 'react';

interface ScreenerCard {
  id: string;
  path: string;
  icon: string;
  title: string;
  description: string;
  badge?: {
    text: string;
    type: 'success' | 'warning' | 'danger' | 'info';
  };
}

const screenerCards: ScreenerCard[] = [
  {
    id: 'box-range',
    path: '/box-range',
    icon: '📦',
    title: '박스권 스크리너',
    description: '2개월 이상 횡보 중인 종목',
    badge: { text: '기초', type: 'info' },
  },
  {
    id: 'box-breakout',
    path: '/box-breakout',
    icon: '🚀',
    title: '박스권 돌파 (거래량 동반)',
    description: '거래량 2배 동반 저항선 돌파',
    badge: { text: '1차 매수', type: 'success' },
  },
  {
    id: 'box-breakout-simple',
    path: '/box-breakout-simple',
    icon: '📈',
    title: '박스권 돌파 (거래량 무관)',
    description: '거래량 무관 돌파 후 10일 이내',
  },
  {
    id: 'pullback',
    path: '/pullback',
    icon: '🎯',
    title: '풀백 스크리너',
    description: '돌파 후 되돌림 종목',
    badge: { text: '2차 매수', type: 'success' },
  },
  {
    id: 'volume-dry-up',
    path: '/volume-dry-up',
    icon: '🏜️',
    title: '거래량 급감 눌림목',
    description: '급등 후 매도세 고갈',
    badge: { text: '세력 보유', type: 'warning' },
  },
  {
    id: 'volume-explosion',
    path: '/volume-explosion',
    icon: '💥',
    title: '거래량 폭발',
    description: '당일 거래량 6배 급등',
    badge: { text: '당일', type: 'danger' },
  },
  {
    id: 'sector-stage',
    path: '/sector-stage',
    icon: '🏭',
    title: '업종별 4단계',
    description: '업종 추세 분석',
  },
];

function Home() {
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  useEffect(() => {
    // 스크리너 결과 파일에서 마지막 업데이트 시간 가져오기
    const fetchLastUpdated = async () => {
      try {
        const response = await fetch('/data/box_range.json');
        if (response.ok) {
          const data = await response.json();
          if (data.meta?.lastUpdated) {
            const date = new Date(data.meta.lastUpdated);
            setLastUpdated(date.toLocaleDateString('ko-KR', {
              year: 'numeric',
              month: 'long',
              day: 'numeric',
            }));
          }
        }
      } catch (error) {
        console.error('Failed to fetch last updated:', error);
      }
    };

    fetchLastUpdated();
  }, []);

  return (
    <div className="min-h-screen px-4 py-8 md:px-8 lg:px-16">
      {/* 헤더 */}
      <header className="mb-12 text-center">
        <h1 className="text-3xl md:text-4xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
          주식 종목 발굴 스크리너
        </h1>
        <p className="text-lg mb-2" style={{ color: 'var(--color-text-secondary)' }}>
          가치투자를 위한 기술적 분석 도구
        </p>
        {lastUpdated && (
          <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
            마지막 업데이트: {lastUpdated}
          </p>
        )}
      </header>

      {/* 스크리너 카드 그리드 */}
      <main className="max-w-4xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
          {screenerCards.map((card) => (
            <Link
              key={card.id}
              to={card.path}
              className="card block p-6 cursor-pointer"
            >
              <div className="flex items-start gap-4">
                {/* 아이콘 */}
                <div className="text-4xl flex-shrink-0">
                  {card.icon}
                </div>

                {/* 내용 */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                      {card.title}
                    </h2>
                    {card.badge && (
                      <span className={`badge badge-${card.badge.type}`}>
                        {card.badge.text}
                      </span>
                    )}
                  </div>
                  <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    {card.description}
                  </p>
                </div>

                {/* 화살표 */}
                <div className="flex-shrink-0 text-xl" style={{ color: 'var(--color-text-muted)' }}>
                  →
                </div>
              </div>
            </Link>
          ))}
        </div>
      </main>

      {/* 푸터 */}
      <footer className="mt-16 text-center">
        <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
          데이터 출처: KRX (한국거래소) · 매일 18:00 자동 업데이트
        </p>
        <p className="text-xs mt-2" style={{ color: 'var(--color-text-muted)' }}>
          투자의 책임은 본인에게 있습니다.
        </p>
      </footer>
    </div>
  );
}

export default Home;
