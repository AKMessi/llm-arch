import React from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage: string;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, errorMessage: '' };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      errorMessage: error?.message || 'Unexpected rendering error.',
    };
  }

  handleReset = () => {
    try {
      window.location.hash = '';
      localStorage.removeItem('architecton_saved_vault_v1');
    } catch {
      // ignore storage errors
    }
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          className="h-screen w-screen flex items-center justify-center p-6 blueprint-grid"
          style={{ color: 'var(--ink-primary)' }}
        >
          <div
            className="max-w-md w-full rounded-xl border p-6 space-y-4"
            style={{
              backgroundColor: 'var(--bg-panel)',
              borderColor: 'var(--border-strong)',
              boxShadow: 'var(--shadow-float)',
            }}
          >
            <div className="space-y-1">
              <div
                className="text-[11px] font-mono-num font-bold uppercase tracking-wider"
                style={{ color: 'var(--stage-output)' }}
              >
                Studio Recovery
              </div>
              <h1 className="font-display font-bold text-[18px]">
                Blueprint state encountered an error
              </h1>
              <p className="text-[12.5px] leading-relaxed" style={{ color: 'var(--ink-secondary)' }}>
                {this.state.errorMessage}
              </p>
            </div>
            <button
              type="button"
              onClick={this.handleReset}
              className="w-full py-2 rounded-md text-[12.5px] font-semibold cursor-pointer"
              style={{
                backgroundColor: 'var(--ink-primary)',
                color: 'var(--ink-inverse)',
              }}
            >
              Reset Blueprint &amp; Reload
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
