import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  currentDose: number | null;
  finalDose: number | null;
  currentRound: number;
  maxRounds: number;
  isRunning: boolean;
  converged: boolean;
  speed: number;
  onSpeedChange: (speed: number) => void;
  onSeek?: (round: number) => void;
}

export function DoseRecommendation({
  currentDose,
  finalDose,
  currentRound,
  maxRounds,
  isRunning,
  converged,
  speed,
  onSpeedChange,
  onSeek,
}: Props) {
  const progress = (currentRound / maxRounds) * 100;
  const displayDose = converged && finalDose !== null ? finalDose : currentDose;

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (onSeek) {
      const value = parseInt(e.target.value, 10);
      onSeek(value);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      style={{
        background: 'linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%)',
        borderRadius: '1rem',
        padding: '1.5rem 2rem',
        color: 'white',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.2)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div style={{ textAlign: 'center', flex: '1', minWidth: '250px' }}>
          <div style={{ fontSize: '0.85rem', opacity: 0.9, marginBottom: '0.5rem' }}>
            {converged ? '✓ 学习完成' : isRunning ? '学习进行中' : '等待开始'}
          </div>
          <AnimatePresence mode="wait">
            {displayDose !== null && (
              <motion.div
                key={displayDose.toFixed(2)}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                style={{ fontSize: '2.5rem', fontWeight: 700, lineHeight: 1 }}
              >
                {displayDose.toFixed(2)} <span style={{ fontSize: '1.2rem', fontWeight: 400 }}>D</span>
              </motion.div>
            )}
          </AnimatePresence>
          <div style={{ fontSize: '0.8rem', opacity: 0.8, marginTop: '0.5rem' }}>
            舒适区间: 3.5 ~ 5.0 D
          </div>
        </div>

        <div style={{ flex: '2', minWidth: '300px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.5rem', opacity: 0.9 }}>
            <span>学习进度 (可拖动)</span>
            <span>{currentRound} / {maxRounds} 轮</span>
          </div>
          <div style={{ position: 'relative', width: '100%', height: '24px', display: 'flex', alignItems: 'center' }}>
            <input
              type="range"
              min={0}
              max={maxRounds}
              value={currentRound}
              onChange={handleSeek}
              disabled={isRunning}
              style={{
                position: 'absolute',
                width: '100%',
                height: '12px',
                WebkitAppearance: 'none',
                appearance: 'none',
                background: 'transparent',
                cursor: isRunning ? 'not-allowed' : 'pointer',
                zIndex: 2,
              }}
            />
            <div
              style={{
                position: 'absolute',
                width: '100%',
                height: '12px',
                background: 'rgba(255, 255, 255, 0.2)',
                borderRadius: '6px',
                overflow: 'hidden',
                pointerEvents: 'none',
              }}
            >
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.3 }}
                style={{
                  height: '100%',
                  background: converged ? '#2ecc71' : 'linear-gradient(90deg, #3498db, #2ecc71)',
                  borderRadius: '6px',
                }}
              />
            </div>
          </div>
          <style>{`
            input[type="range"]::-webkit-slider-thumb {
              -webkit-appearance: none;
              appearance: none;
              width: 20px;
              height: 20px;
              border-radius: 50%;
              background: white;
              cursor: pointer;
              box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
              border: 2px solid #2d5a87;
              margin-top: -4px;
            }
            input[type="range"]::-moz-range-thumb {
              width: 20px;
              height: 20px;
              border-radius: 50%;
              background: white;
              cursor: pointer;
              box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
              border: 2px solid #2d5a87;
            }
            input[type="range"]:disabled::-webkit-slider-thumb {
              cursor: not-allowed;
              opacity: 0.5;
            }
            input[type="range"]:disabled::-moz-range-thumb {
              cursor: not-allowed;
              opacity: 0.5;
            }
          `}</style>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.85rem', opacity: 0.9 }}>播放速度</span>
          <div style={{ display: 'flex', gap: '0.3rem' }}>
            {[1, 2, 5].map((s) => (
              <button
                key={s}
                onClick={() => onSpeedChange(s)}
                style={{
                  padding: '0.4rem 0.8rem',
                  background: speed === s ? 'rgba(255, 255, 255, 0.3)' : 'transparent',
                  border: '1px solid rgba(255, 255, 255, 0.5)',
                  borderRadius: '0.3rem',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '0.8rem',
                  transition: 'all 0.2s',
                }}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {converged && finalDose !== null && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            textAlign: 'center',
            marginTop: '1rem',
            padding: '0.8rem',
            background: 'rgba(46, 204, 113, 0.2)',
            borderRadius: '0.5rem',
            fontWeight: 600,
          }}
        >
          🎉 推荐剂量 = {finalDose.toFixed(2)} D (舒适区间内)
        </motion.div>
      )}
    </motion.div>
  );
}