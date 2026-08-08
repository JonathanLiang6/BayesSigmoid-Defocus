import { useState } from 'react';
import { LearningStep } from '../types';
import { PosteriorDistributionChart } from './PosteriorDistributionChart';
import { ResidualDiagnosticsChart } from './ResidualDiagnosticsChart';
import { ObservedVsPredictedChart } from './ObservedVsPredictedChart';
import { RecommendedDoseDistributionChart } from './RecommendedDoseDistributionChart';

interface Props {
  steps: LearningStep[];
  finalDose: number | null;
}

type ChartType = 'posterior' | 'residual' | 'observed' | 'distribution';

const chartOptions: { key: ChartType; label: string }[] = [
  { key: 'posterior', label: '后验分布' },
  { key: 'residual', label: '残差诊断' },
  { key: 'observed', label: '观测vs预测' },
  { key: 'distribution', label: '剂量分布' },
];

export function ExtendedVisualization({ steps, finalDose }: Props) {
  const [selectedChart, setSelectedChart] = useState<ChartType>('posterior');

  return (
    <div style={{
      background: 'white',
      borderRadius: '1rem',
      padding: '1rem',
      boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
      height: '100%',
    }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '0.5rem'
      }}>
        <h3 style={{ margin: 0, color: '#1e3a5f', fontSize: '1rem' }}>
          扩展可视化
        </h3>
        <div style={{ 
          display: 'flex', 
          gap: '0.3rem',
          flexWrap: 'wrap',
          justifyContent: 'flex-end'
        }}>
          {chartOptions.map((option) => (
            <button
              key={option.key}
              onClick={() => setSelectedChart(option.key)}
              style={{
                padding: '0.4rem 0.8rem',
                fontSize: '0.75rem',
                background: selectedChart === option.key ? '#2d5a87' : '#f0f0f0',
                color: selectedChart === option.key ? 'white' : '#555',
                border: 'none',
                borderRadius: '0.3rem',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
      
      <div style={{ height: 'calc(100% - 60px)' }}>
        {selectedChart === 'posterior' && <PosteriorDistributionChart steps={steps} />}
        {selectedChart === 'residual' && <ResidualDiagnosticsChart steps={steps} />}
        {selectedChart === 'observed' && <ObservedVsPredictedChart steps={steps} />}
        {selectedChart === 'distribution' && (
          <RecommendedDoseDistributionChart steps={steps} finalDose={finalDose} />
        )}
      </div>
    </div>
  );
}