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

const iconSvgMap: Record<string, ReactNode> = {
  '📦': <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>,
  '🚀': <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="m4.5 16.5 2-2a5 5 0 0 1 3.5-1.5h4a5 5 0 0 1 3.5 1.5l2 2"/><path d="M5.5 6.5 12 2l6.5 4.5"/><path d="M12 2v6"/></svg>,
  '📈': <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>,
  '🎯': <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>,
  '🏜️': <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>,
  '💥': <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
  '🏭': <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>,
};

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
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M15 18l-6-6 6-6"/>
        </svg>
        홈으로
      </Link>

      {/* 헤더 */}
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-2 flex-wrap">
          <span className="flex items-center" style={{ color: 'var(--color-accent)' }}>{iconSvgMap[icon] || icon}</span>
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
