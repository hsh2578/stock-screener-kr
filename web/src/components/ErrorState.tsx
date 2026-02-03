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
      <div className="text-4xl mb-4">⚠️</div>
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
