import { useState, useMemo, useEffect } from 'react';
import { Header } from './components/Header';
import { ParameterPanel } from './components/ParameterPanel';
import { LearningProcessChart } from './components/LearningProcessChart';
import { LearningEffectChart } from './components/LearningEffectChart';
import { ExtendedVisualization } from './components/ExtendedVisualization';
import { DoseRecommendation } from './components/DoseRecommendation';
import { useSimulation } from './hooks/useSimulation';
import { PatientParams, PatientInfo } from './types';

const initialParams: PatientParams = {
  age: 22,
  sex: 1,
  axialLength: 25.5,
  choroidalThickness: 10,
  cvi: 0.45,
  eyeSide: 0,
  rdv15: 0,
  refractionDefocus: 0,
};

export default function App() {
  const [params, setParams] = useState<PatientParams>(initialParams);
  const [patients, setPatients] = useState<PatientInfo[]>([]);
  const [loadingPatients, setLoadingPatients] = useState(false);
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  
  const {
    isRunning,
    isPaused,
    isLoading,
    currentRound,
    maxRounds,
    speed,
    steps,
    finalDose,
    converged,
    start,
    pause,
    resume,
    reset,
    setSpeed,
    seek,
  } = useSimulation(30);

  useEffect(() => {
    const fetchPatients = async () => {
      setLoadingPatients(true);
      try {
        const response = await fetch('http://localhost:8000/api/patients');
        if (response.ok) {
          const data = await response.json();
          setPatients(data.patients);
        }
      } catch (error) {
        console.error('Failed to fetch patients:', error);
      } finally {
        setLoadingPatients(false);
      }
    };
    fetchPatients();
  }, []);

  const handlePatientSelect = async (patientId: string) => {
    setSelectedPatientId(patientId);
    try {
      const response = await fetch(`http://localhost:8000/api/patients/${patientId}`);
      if (response.ok) {
        const patient = await response.json();
        setParams({
          age: patient.age,
          sex: patient.gender,
          axialLength: patient.axial_length,
          choroidalThickness: patient.choroid_thickness,
          cvi: patient.cvi,
          eyeSide: patient.eye === '左眼' ? 0 : 1,
          rdv15: patient.rdv15,
          refractionDefocus: patient.refractive_error,
        });
      }
    } catch (error) {
      console.error('Failed to fetch patient:', error);
    }
  };

  const displayedSteps = useMemo(() => {
    return steps.slice(0, currentRound);
  }, [steps, currentRound]);

  const currentStep = useMemo(() => {
    if (currentRound > 0 && steps.length > 0) {
      return steps[Math.min(currentRound - 1, steps.length - 1)];
    }
    return null;
  }, [steps, currentRound]);

  const handleStart = () => {
    if (selectedPatientId) {
      start(selectedPatientId);
    }
  };

  const handleReset = () => {
    reset();
  };

  const currentDose = currentStep?.dose ?? null;
  const currentAccuracy = currentStep?.accuracy ?? null;
  const currentMae = currentStep?.mae ?? null;
  const currentUncertainty = currentStep 
    ? (currentStep.upperBound - currentStep.lowerBound) / 2 
    : null;

  const statusText = isLoading ? '加载中...' : (isRunning ? (isPaused ? '已暂停' : '运行中') : (converged ? '已完成' : '等待'));

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(180deg, #f5f7fa 0%, #e4e8ec 100%)',
      padding: '1rem',
      paddingBottom: '2rem',
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        <Header />

        <ParameterPanel
          params={params}
          onChange={setParams}
          onStart={handleStart}
          onReset={handleReset}
          isRunning={isRunning || isLoading}
          disabled={isRunning || isLoading || !selectedPatientId}
          selectorDisabled={isRunning || isLoading}
          loadingPatients={loadingPatients}
          patients={patients}
          selectedPatientId={selectedPatientId}
          onPatientSelect={handlePatientSelect}
        />

        <DoseRecommendation
          currentDose={currentDose}
          finalDose={finalDose}
          currentRound={currentRound}
          maxRounds={maxRounds}
          isRunning={isRunning}
          converged={converged}
          speed={speed}
          onSpeedChange={setSpeed}
          onSeek={seek}
        />

        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr',
          gap: '1.5rem',
          marginTop: '1.5rem',
        }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1.5rem' }}>
            <LearningEffectChart steps={displayedSteps} />
            <div>
              <LearningProcessChart
                steps={displayedSteps}
                converged={converged}
                finalDose={finalDose}
              />
            </div>
          </div>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
          gap: '1.5rem',
          marginTop: '1.5rem',
        }}>
          <ExtendedVisualization steps={displayedSteps} finalDose={finalDose} />
          <div style={{
            background: 'white',
            borderRadius: '1rem',
            padding: '1rem',
            boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
          }}>
            <h3 style={{ margin: '0 0 0.5rem 0', color: '#1e3a5f', fontSize: '1rem' }}>
              实时状态监控
            </h3>
            <p style={{ margin: '0 0 1rem 0', color: '#777', fontSize: '0.8rem' }}>
              当前模拟状态与控制选项
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.8rem' }}>
              <div style={{ padding: '0.8rem', background: '#f8f9fa', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.7rem', color: '#777', marginBottom: '0.2rem' }}>状态</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: isRunning || isLoading ? '#2d5a87' : '#555' }}>
                  {statusText}
                </div>
              </div>
              <div style={{ padding: '0.8rem', background: '#f8f9fa', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.7rem', color: '#777', marginBottom: '0.2rem' }}>当前轮次</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#2d5a87' }}>{currentRound} / {maxRounds}</div>
              </div>
              <div style={{ padding: '0.8rem', background: '#f8f9fa', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.7rem', color: '#777', marginBottom: '0.2rem' }}>收敛状态</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: converged ? '#27ae60' : '#e74c3c' }}>
                  {converged ? '已收敛' : '收敛中'}
                </div>
              </div>
              <div style={{ padding: '0.8rem', background: '#f8f9fa', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.7rem', color: '#777', marginBottom: '0.2rem' }}>准确率</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#2d5a87' }}>
                  {currentAccuracy !== null ? `${(currentAccuracy * 100).toFixed(1)}%` : '-'}
                </div>
              </div>
              <div style={{ padding: '0.8rem', background: '#f8f9fa', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.7rem', color: '#777', marginBottom: '0.2rem' }}>MAE</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#e74c3c' }}>
                  {currentMae !== null ? currentMae.toFixed(4) : '-'}
                </div>
              </div>
              <div style={{ padding: '0.8rem', background: '#f8f9fa', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.7rem', color: '#777', marginBottom: '0.2rem' }}>不确定性</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#9b59b6' }}>
                  {currentUncertainty !== null ? `±${currentUncertainty.toFixed(3)}` : '-'}
                </div>
              </div>
              <div style={{ padding: '0.8rem', background: '#f8f9fa', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.7rem', color: '#777', marginBottom: '0.2rem' }}>推荐剂量</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#2d5a87' }}>
                  {currentDose !== null ? `${currentDose.toFixed(2)} D` : '-'}
                </div>
              </div>
              <div style={{ padding: '0.8rem', background: '#f8f9fa', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.7rem', color: '#777', marginBottom: '0.2rem' }}>数据点数</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#2d5a87' }}>{displayedSteps.length}</div>
              </div>
            </div>
            {isRunning && (
              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
                <button
                  onClick={isPaused ? resume : pause}
                  style={{
                    flex: 1,
                    padding: '0.6rem',
                    background: 'linear-gradient(135deg, #2d5a87 0%, #1e3a5f 100%)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '0.5rem',
                    fontSize: '0.9rem',
                    cursor: 'pointer',
                  }}
                >
                  {isPaused ? '继续' : '暂停'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}