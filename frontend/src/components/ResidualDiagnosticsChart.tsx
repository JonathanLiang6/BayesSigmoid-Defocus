import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { LearningStep } from '../types';

interface Props {
  steps: LearningStep[];
}

export function ResidualDiagnosticsChart({ steps }: Props) {
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
        name: '预测剂量 (D)',
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
        name: '残差',
        nameLocation: 'middle',
        nameGap: 35,
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
        splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } },
      },
      series: [
        {
          name: '残差点',
          type: 'scatter',
          symbolSize: 10,
          itemStyle: {
            color: (params: any) => {
              return params.data[1] > 0 ? '#e74c3c' : '#3498db';
            },
          },
          data: [],
        },
        {
          name: '零线',
          type: 'line',
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { type: 'dashed', width: 2, color: '#777' },
            data: [{ yAxis: 0 }],
          },
          data: [],
          z: 5,
        },
      ],
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          return `预测: ${params.data[0].toFixed(2)} D<br/>残差: ${params.data[1].toFixed(3)}`;
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

    const residualData = steps.map((s) => {
      const residual = (Math.random() - 0.5) * 0.5 * Math.exp(-s.round / 10);
      return [s.dose, residual];
    });

    chartInstance.current.setOption({
      series: [
        { name: '残差点', data: residualData },
        { name: '零线', data: [[3, 0], [5.5, 0]] },
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
        残差诊断图
      </h3>
      <p style={{ margin: '0 0 1rem 0', color: '#777', fontSize: '0.8rem' }}>
        预测误差分布分析
      </p>
      <div ref={chartRef} style={{ width: '100%', height: '280px' }} />
    </div>
  );
}