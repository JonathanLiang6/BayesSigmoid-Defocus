import { motion } from 'framer-motion';

interface Props {
  currentRound: number;
  maxRounds: number;
  converged: boolean;
}

export function LearningProgressBar({ currentRound, maxRounds, converged }: Props) {
  const progress = (currentRound / maxRounds) * 100;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      style={{
        marginTop: '1rem',
        padding: '1rem',
        background: 'white',
        borderRadius: '0.8rem',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
      }}
    >
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '0.5rem' 
      }}>
        <span style={{ fontSize: '0.85rem', color: '#555', fontWeight: 500 }}>
          学习进度
        </span>
        <span style={{ 
          fontSize: '0.85rem', 
          color: converged ? '#27ae60' : '#2d5a87',
          fontWeight: 600 
        }}>
          {currentRound} / {maxRounds} 轮
          {converged && ' ✓'}
        </span>
      </div>
      <div style={{
        width: '100%',
        height: '8px',
        background: '#e8ecf0',
        borderRadius: '4px',
        overflow: 'hidden',
      }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3 }}
          style={{
            height: '100%',
            background: converged 
              ? 'linear-gradient(90deg, #27ae60, #2ecc71)' 
              : 'linear-gradient(90deg, #3498db, #2ecc71)',
            borderRadius: '4px',
          }}
        />
      </div>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginTop: '0.5rem',
        fontSize: '0.75rem',
        color: '#777',
      }}>
        <span>初始不确定性高</span>
        <span>收敛完成</span>
      </div>
    </motion.div>
  );
}