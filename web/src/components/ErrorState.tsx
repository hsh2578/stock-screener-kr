interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

function ErrorState({
  message = '데이터를 불러오지 못했습니다.',
  onRetry,
}: ErrorStateProps) {
  return (
    <div className="card p-8 text-center">
      <div className="mb-4 flex justify-center">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
      </div>
      <p className="text-lg mb-2" style={{ color: 'var(--color-text-primary)' }}>
        {message}
      </p>
      <p className="text-sm mb-4" style={{ color: 'var(--color-text-muted)' }}>
        잠시 후 다시 시도해주세요.
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          style={{
            backgroundColor: 'var(--color-accent)',
            color: 'white',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--color-accent-hover)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--color-accent)';
          }}
        >
          새로고침
        </button>
      )}
    </div>
  );
}

export default ErrorState;
