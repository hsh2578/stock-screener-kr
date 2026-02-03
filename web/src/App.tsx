import { HashRouter as Router, Routes, Route } from 'react-router-dom';

// 페이지 컴포넌트
import Home from './pages/Home';
import BoxRange from './pages/BoxRange';
import BoxBreakout from './pages/BoxBreakout';
import BoxBreakoutSimple from './pages/BoxBreakoutSimple';
import Pullback from './pages/Pullback';
import VolumeDryUp from './pages/VolumeDryUp';
import VolumeExplosion from './pages/VolumeExplosion';
import SectorStage from './pages/SectorStage';

function App() {
  return (
    <Router>
      <Routes>
        {/* 메인 페이지 */}
        <Route path="/" element={<Home />} />

        {/* 스크리너 페이지 */}
        <Route path="/box-range" element={<BoxRange />} />
        <Route path="/box-breakout" element={<BoxBreakout />} />
        <Route path="/box-breakout-simple" element={<BoxBreakoutSimple />} />
        <Route path="/pullback" element={<Pullback />} />
        <Route path="/volume-dry-up" element={<VolumeDryUp />} />
        <Route path="/volume-explosion" element={<VolumeExplosion />} />
        <Route path="/sector-stage" element={<SectorStage />} />

        {/* 404 페이지 */}
        <Route
          path="*"
          element={
            <div className="min-h-screen flex items-center justify-center">
              <div className="text-center">
                <h1 className="text-4xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
                  404
                </h1>
                <p className="mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                  페이지를 찾을 수 없습니다.
                </p>
                <a href="/" className="text-sm">
                  홈으로 돌아가기 →
                </a>
              </div>
            </div>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
