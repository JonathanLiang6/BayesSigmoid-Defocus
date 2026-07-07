import { motion } from 'framer-motion';

export function Header() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      style={{
        background: 'linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%)',
        color: 'white',
        padding: '1.5rem 2rem',
        borderRadius: '0 0 1rem 1rem',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
        marginBottom: '1.5rem',
      }}
    >
      <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 600 }}>
        离焦剂量-脉络膜反应贝叶斯分析系统
      </h1>
      <p style={{ margin: '0.5rem 0 0', opacity: 0.9, fontSize: '0.9rem' }}>
        基于贝叶斯主动学习的个性化剂量优化与可视化
      </p>
    </motion.header>
  );
}