import { Link } from 'react-router-dom';
import type { ReactNode } from 'react';

interface PageLayoutProps {
  icon: string;
  title: string;
  description: string;
  badge?: {
    text: string;
    type: 'success' | 'warning' | 'danger' | 'info';
  };
  lastUpdated?: string;
  totalCount?: number;
  screenedFrom?: number;
  children: ReactNode;
}

function PageLayout({
  icon,
  title,
  description,
  badge,
  lastUpdated,
  totalCount,
  screenedFrom,
  children,
}: PageLayoutProps) {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="min-h-screen px-4 py-8 md:px-8 lg:px-16">
      {/* 뒤로가기 */}
      <Link
        to="/"
        className="inline-flex items-center gap-2 mb-8 text-sm hover:opacity-80 transition-opacity"
        style={{ color: 'var(--color-text-secondary)' }}
      >
        ← 홈으로
      </Link>

      {/* 헤더 */}
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-2 flex-wrap">
          <span className="text-3xl">{icon}</span>
          <h1
            className="text-2xl md:text-3xl font-bold"
            style={{ color: 'var(--color-text-primary)' }}
          >
            {title}
          </h1>
          {badge && (
            <span className={`badge badge-${badge.type}`}>{badge.text}</span>
          )}
        </div>
        <p style={{ color: 'var(--color-text-secondary)' }}>{description}</p>

        {/* 메타 정보 */}
        <div className="flex flex-wrap gap-4 mt-4 text-sm">
          {lastUpdated && (
            <span style={{ color: 'var(--color-text-muted)' }}>
              데이터 기준: {formatDate(lastUpdated)}
            </span>
          )}
          {totalCount !== undefined && (
            <span style={{ color: 'var(--color-accent)' }}>
              {totalCount}개 종목
              {screenedFrom && (
                <span style={{ color: 'var(--color-text-muted)' }}>
                  {' '}
                  / {screenedFrom.toLocaleString()}개 중
                </span>
              )}
            </span>
          )}
        </div>
      </header>

      {/* 컨텐츠 영역 */}
      <main>{children}</main>
    </div>
  );
}

export default PageLayout;
