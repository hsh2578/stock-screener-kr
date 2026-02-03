interface LoadingStateProps {
  type?: 'table' | 'card' | 'full';
  rows?: number;
}

function LoadingState({ type = 'table', rows = 5 }: LoadingStateProps) {
  if (type === 'full') {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-center">
          <div
            className="inline-block w-10 h-10 border-2 rounded-full animate-spin"
            style={{
              borderColor: 'var(--color-border)',
              borderTopColor: 'var(--color-accent)',
            }}
          />
          <p className="mt-4 text-sm" style={{ color: 'var(--color-text-muted)' }}>
            데이터를 불러오는 중...
          </p>
        </div>
      </div>
    );
  }

  if (type === 'card') {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Array.from({ length: rows }).map((_, i) => (
          <div
            key={i}
            className="card p-4 animate-pulse"
            style={{ backgroundColor: 'var(--color-bg-secondary)' }}
          >
            <div
              className="h-6 rounded mb-3"
              style={{ backgroundColor: 'var(--color-bg-tertiary)', width: '60%' }}
            />
            <div
              className="h-4 rounded mb-2"
              style={{ backgroundColor: 'var(--color-bg-tertiary)', width: '40%' }}
            />
            <div
              className="h-4 rounded"
              style={{ backgroundColor: 'var(--color-bg-tertiary)', width: '80%' }}
            />
          </div>
        ))}
      </div>
    );
  }

  // Table skeleton
  return (
    <div className="overflow-hidden rounded-lg" style={{ border: '1px solid var(--color-border)' }}>
      {/* Header */}
      <div
        className="flex gap-4 p-4 animate-pulse"
        style={{ backgroundColor: 'var(--color-bg-secondary)' }}
      >
        {[1, 2, 3, 4, 5].map((i) => (
          <div
            key={i}
            className="h-4 rounded"
            style={{
              backgroundColor: 'var(--color-bg-tertiary)',
              flex: i === 1 ? 2 : 1,
            }}
          />
        ))}
      </div>

      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div
          key={rowIndex}
          className="flex gap-4 p-4 animate-pulse"
          style={{
            borderTop: '1px solid var(--color-border)',
          }}
        >
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="h-4 rounded"
              style={{
                backgroundColor: 'var(--color-bg-tertiary)',
                flex: i === 1 ? 2 : 1,
                opacity: 0.5 + (rowIndex % 3) * 0.2,
              }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export default LoadingState;
