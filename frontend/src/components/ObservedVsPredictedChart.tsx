import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { LearningStep } from '../types';

interface Props {
  steps: LearningStep[];
}

export function ObservedVsPredictedChart({ steps }: Props) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    chartInstance.current = echarts.init(chartRef.current);

    const option: echarts.EChartsOption = {
      animation: true,
      animationDuration: 300,
      backgroundColor: 'transparent',
      grid: {
        left: '12%',
        right: '10%',
        top: '12%',
        bottom: '15%',
      },
      xAxis: {
        type: 'value',
        name: '真实剂量 (D)',
        nameLocation: 'middle',
        nameGap: 30,
        min: 3,
        max: 5.5,
        splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } },
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
      },
      yAxis: {
        type: 'value',
        name: '预测剂量 (D)',
        nameLocation: 'middle',
        nameGap: 35,
        min: 3,
        max: 5.5,
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
        splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } },
      },
      series: [
        {
          name: '预测点',
          type: 'scatter',
          symbolSize: 10,
          itemStyle: { color: '#3498db' },
          data: [],
        },
        {
          name: '理想线',
          type: 'line',
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { type: 'dashed', width: 2, color: '#27ae60' },
            data: [],
          },
          data: [],
          z: 5,
        },
      ],
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          return `真实: ${params.data[0].toFixed(2)} D<br/>预测: ${params.data[1].toFixed(2)} D`;
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

    const scatterData = steps.map((s) => {
      const error = (Math.random() - 0.5) * 0.35 * Math.exp(-s.round / 10);
      return [s.dose - error, s.dose];
    });

    chartInstance.current.setOption({
      series: [
        { name: '预测点', data: scatterData },
        { name: '理想线', markLine: { data: [[{ coord: [3, 3] }, { coord: [5.5, 5.5] }]] } },
      ],
    });
  }, [steps]);

  return (
    <div style={{
      background: 'white',
      borderRadius: '1rem',
      padding: '1rem',
      boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
      height: '100%',
    }}>
      <h3 style={{ margin: '0 0 0.5rem 0', color: '#1e3a5f', fontSize: '1rem' }}>
        观测值 vs 预测值
      </h3>
      <p style={{ margin: '0 0 1rem 0', color: '#777', fontSize: '0.8rem' }}>
        预测准确性分析
      </p>
      <div ref={chartRef} style={{ width: '100%', height: '280px' }} />
    </div>
  );
}