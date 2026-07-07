import { useState, useEffect } from 'react';
import { PatientParams, PatientInfo } from '../types';

interface Props {
  params: PatientParams;
  onChange: (params: PatientParams) => void;
  onStart: () => void;
  onReset: () => void;
  isRunning: boolean;
  disabled: boolean;
  selectorDisabled?: boolean;
  loadingPatients?: boolean;
  patients?: PatientInfo[];
  selectedPatientId?: string | null;
  onPatientSelect?: (patientId: string) => void;
}

export function ParameterPanel({
  params,
  onChange,
  onStart,
  onReset,
  isRunning,
  disabled,
  selectorDisabled = false,
  loadingPatients = false,
  patients = [],
  selectedPatientId = null,
  onPatientSelect,
}: Props) {
  const handleNumberChange = (field: keyof PatientParams, value: number) => {
    onChange({ ...params, [field]: value });
  };

  const handlePatientChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const patientId = e.target.value;
    if (onPatientSelect && patientId) {
      onPatientSelect(patientId);
    }
  };

  return (
    <div style={{
      background: 'white',
      borderRadius: '1rem',
      padding: '1.5rem',
      boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
      marginBottom: '1.5rem',
    }}>
      <h3 style={{ margin: '0 0 1rem 0', color: '#1e3a5f', fontSize: '1rem' }}>患者参数输入</h3>
      
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.25rem', color: '#555', fontSize: '0.75rem' }}>
          选择患者
        </label>
        <select
          value={selectedPatientId ?? ''}
          onChange={handlePatientChange}
          disabled={selectorDisabled || loadingPatients}
          style={{
            width: '100%',
            padding: '0.5rem',
            border: '1px solid #ddd',
            borderRadius: '0.3rem',
            fontSize: '0.85rem',
            outline: 'none',
            background: 'white',
            cursor: selectorDisabled ? 'not-allowed' : 'pointer',
          }}
        >
          <option value="">-- 请选择患者 --</option>
          {loadingPatients && <option value="">加载中...</option>}
          {patients.map((p) => (
            <option key={p.id} value={p.id}>
              患者{p.subject_id} - {p.eye} ({p.batch})
            </option>
          ))}
        </select>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.8rem' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '0.25rem', color: '#555', fontSize: '0.75rem' }}>
            眼别
          </label>
          <div style={{ display: 'flex', gap: '0.25rem' }}>
            <button
              onClick={() => handleNumberChange('eyeSide', 0)}
              disabled={disabled}
              style={{
                flex: 1,
                padding: '0.4rem 0.2rem',
                border: params.eyeSide === 0 ? '2px solid #2d5a87' : '1px solid #ddd',
                borderRadius: '0.3rem',
                background: params.eyeSide === 0 ? '#e8f0f8' : 'white',
                cursor: disabled ? 'not-allowed' : 'pointer',
                fontSize: '0.8rem',
                fontWeight: params.eyeSide === 0 ? 600 : 400,
              }}
            >
              左眼 OS
            </button>
            <button
              onClick={() => handleNumberChange('eyeSide', 1)}
              disabled={disabled}
              style={{
                flex: 1,
                padding: '0.4rem 0.2rem',
                border: params.eyeSide === 1 ? '2px solid #2d5a87' : '1px solid #ddd',
                borderRadius: '0.3rem',
                background: params.eyeSide === 1 ? '#e8f0f8' : 'white',
                cursor: disabled ? 'not-allowed' : 'pointer',
                fontSize: '0.8rem',
                fontWeight: params.eyeSide === 1 ? 600 : 400,
              }}
            >
              右眼 OD
            </button>
          </div>
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.25rem', color: '#555', fontSize: '0.75rem' }}>
            性别
          </label>
          <div style={{ display: 'flex', gap: '0.25rem' }}>
            <button
              onClick={() => handleNumberChange('sex', 0)}
              disabled={disabled}
              style={{
                flex: 1,
                padding: '0.4rem 0.2rem',
                border: params.sex === 0 ? '2px solid #2d5a87' : '1px solid #ddd',
                borderRadius: '0.3rem',
                background: params.sex === 0 ? '#e8f0f8' : 'white',
                cursor: disabled ? 'not-allowed' : 'pointer',
                fontSize: '0.8rem',
                fontWeight: params.sex === 0 ? 600 : 400,
              }}
            >
              女
            </button>
            <button
              onClick={() => handleNumberChange('sex', 1)}
              disabled={disabled}
              style={{
                flex: 1,
                padding: '0.4rem 0.2rem',
                border: params.sex === 1 ? '2px solid #2d5a87' : '1px solid #ddd',
                borderRadius: '0.3rem',
                background: params.sex === 1 ? '#e8f0f8' : 'white',
                cursor: disabled ? 'not-allowed' : 'pointer',
                fontSize: '0.8rem',
                fontWeight: params.sex === 1 ? 600 : 400,
              }}
            >
              男
            </button>
          </div>
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.25rem', color: '#555', fontSize: '0.75rem' }}>
            年龄 (岁)
          </label>
          <input
            type="number"
            value={params.age}
            onChange={(e) => handleNumberChange('age', Number(e.target.value))}
            disabled={disabled}
            min={10}
            max={80}
            style={{
              width: '100%',
              padding: '0.4rem',
              border: '1px solid #ddd',
              borderRadius: '0.3rem',
              fontSize: '0.85rem',
              outline: 'none',
            }}
          />
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.25rem', color: '#555', fontSize: '0.75rem' }}>
            眼轴长度 (mm)
          </label>
          <input
            type="number"
            value={params.axialLength}
            onChange={(e) => handleNumberChange('axialLength', Number(e.target.value))}
            disabled={disabled}
            step={0.01}
            min={20}
            max={30}
            style={{
              width: '100%',
              padding: '0.4rem',
              border: '1px solid #ddd',
              borderRadius: '0.3rem',
              fontSize: '0.85rem',
              outline: 'none',
            }}
          />
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.25rem', color: '#555', fontSize: '0.75rem' }}>
            脉络膜厚度 (μm)
          </label>
          <input
            type="number"
            value={params.choroidalThickness}
            onChange={(e) => handleNumberChange('choroidalThickness', Number(e.target.value))}
            disabled={disabled}
            step={0.1}
            min={0}
            max={100}
            style={{
              width: '100%',
              padding: '0.4rem',
              border: '1px solid #ddd',
              borderRadius: '0.3rem',
              fontSize: '0.85rem',
              outline: 'none',
            }}
          />
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.25rem', color: '#555', fontSize: '0.75rem' }}>
            CVI
          </label>
          <input
            type="number"
            value={params.cvi}
            onChange={(e) => handleNumberChange('cvi', Number(e.target.value))}
            disabled={disabled}
            step={0.001}
            min={0}
            max={1}
            style={{
              width: '100%',
              padding: '0.4rem',
              border: '1px solid #ddd',
              borderRadius: '0.3rem',
              fontSize: '0.85rem',
              outline: 'none',
            }}
          />
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.25rem', color: '#555', fontSize: '0.75rem' }}>
            RDV15 (D)
          </label>
          <input
            type="number"
            value={params.rdv15}
            onChange={(e) => handleNumberChange('rdv15', Number(e.target.value))}
            disabled={disabled}
            step={0.01}
            min={-1}
            max={1}
            style={{
              width: '100%',
              padding: '0.4rem',
              border: '1px solid #ddd',
              borderRadius: '0.3rem',
              fontSize: '0.85rem',
              outline: 'none',
            }}
          />
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.25rem', color: '#555', fontSize: '0.75rem' }}>
            验光离焦量 (D)
          </label>
          <input
            type="number"
            value={params.refractionDefocus}
            onChange={(e) => handleNumberChange('refractionDefocus', Number(e.target.value))}
            disabled={disabled}
            step={0.25}
            min={-10}
            max={0}
            style={{
              width: '100%',
              padding: '0.4rem',
              border: '1px solid #ddd',
              borderRadius: '0.3rem',
              fontSize: '0.85rem',
              outline: 'none',
            }}
          />
        </div>
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginTop: '1.2rem', justifyContent: 'center' }}>
        <button
          onClick={onStart}
          disabled={isRunning || disabled}
          style={{
            padding: '0.7rem 2rem',
            background: isRunning ? '#ccc' : 'linear-gradient(135deg, #2d5a87 0%, #1e3a5f 100%)',
            color: 'white',
            border: 'none',
            borderRadius: '0.5rem',
            fontSize: '0.95rem',
            fontWeight: 600,
            cursor: isRunning ? 'not-allowed' : 'pointer',
            boxShadow: isRunning ? 'none' : '0 4px 12px rgba(45, 90, 135, 0.3)',
            transition: 'all 0.3s ease',
          }}
        >
          {isRunning ? '学习中...' : '开始模拟学习'}
        </button>
        <button
          onClick={onReset}
          style={{
            padding: '0.7rem 1.5rem',
            background: 'white',
            color: '#e74c3c',
            border: '2px solid #e74c3c',
            borderRadius: '0.5rem',
            fontSize: '0.95rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.3s ease',
          }}
        >
          重置
        </button>
      </div>
    </div>
  );
}