import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { LearningStep } from '../types';

interface Props {
  steps: LearningStep[];
}

export function PosteriorDistributionChart({ steps }: Props) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    chartInstance.current = echarts.init(chartRef.current);

    const option: echarts.EChartsOption = {
      animation: true,
      animationDuration: 300,
      animationEasing: 'cubicInOut',
      backgroundColor: 'transparent',
      grid: {
        left: '12%',
        right: '10%',
        top: '12%',
        bottom: '15%',
      },
      xAxis: {
        type: 'value',
        name: '离焦剂量 (D)',
        nameLocation: 'middle',
        nameGap: 30,
        min: 3,
        max: 5.5,
        splitLine: {
          lineStyle: { type: 'dashed', color: '#e0e0e0' },
        },
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
      },
      yAxis: {
        type: 'value',
        name: '概率密度',
        nameLocation: 'middle',
        nameGap: 35,
        min: 0,
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
        splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } },
      },
      series: [
        {
          name: '后验分布',
          type: 'line',
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 0 },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(46, 204, 113, 0.6)' },
              { offset: 1, color: 'rgba(46, 204, 113, 0.1)' },
            ]),
          },
          data: [],
        },
        {
          name: '均值线',
          type: 'line',
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { type: 'dashed', width: 2, color: '#e74c3c' },
            label: { formatter: '均值', position: 'end' },
            data: [],
          },
          z: 5,
        },
      ],
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params[0]) return '';
          const p = params[0];
          return `剂量: ${p.data[0].toFixed(2)} D<br/>概率密度: ${p.data[1].toFixed(3)}`;
        },
      },
    };

    chartInstance.current.setOption(option);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, []);

  useEffect(() => {
    if (!chartInstance.current || steps.length === 0) return;

    const currentStep = steps[steps.length - 1];
    const mean = currentStep.dose;
    const std = (currentStep.upperBound - currentStep.lowerBound) / 4;

    const distributionData: [number, number][] = [];
    for (let x = 3; x <= 5.5; x += 0.05) {
      const gaussian = Math.exp(-0.5 * Math.pow((x - mean) / std, 2)) / (std * Math.sqrt(2 * Math.PI));
      distributionData.push([x, gaussian]);
    }

    chartInstance.current.setOption({
      series: [
        { name: '后验分布', data: distributionData },
        {
          name: '均值线',
          markLine: {
            data: [{ xAxis: mean }],
          },
        },
      ],
    });
  }, [steps]);

  const currentStep = steps.length > 0 ? steps[steps.length - 1] : null;
  const std = currentStep ? (currentStep.upperBound - currentStep.lowerBound) / 4 : 0.5;

  return (
    <div style={{
      background: 'white',
      borderRadius: '1rem',
      padding: '1rem',
      boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
      height: '100%',
    }}>
      <h3 style={{ margin: '0 0 0.5rem 0', color: '#1e3a5f', fontSize: '1rem' }}>
        后验分布变化
      </h3>
      <p style={{ margin: '0 0 1rem 0', color: '#777', fontSize: '0.8rem' }}>
        剂量分布从扁平逐渐变为尖锐（不确定性降低）
      </p>
      <div ref={chartRef} style={{ width: '100%', height: '280px' }} />
      {currentStep && (
        <div style={{
          display: 'flex',
          justifyContent: 'space-around',
          marginTop: '1rem',
          padding: '0.8rem',
          background: '#f8f9fa',
          borderRadius: '0.5rem',
          fontSize: '0.85rem',
        }}>
          <div>
            <span style={{ color: '#777' }}>均值: </span>
            <span style={{ fontWeight: 600, color: '#2d5a87' }}>{currentStep.dose.toFixed(2)} D</span>
          </div>
          <div>
            <span style={{ color: '#777' }}>标准差: </span>
            <span style={{ fontWeight: 600, color: '#27ae60' }}>{std.toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  );
}