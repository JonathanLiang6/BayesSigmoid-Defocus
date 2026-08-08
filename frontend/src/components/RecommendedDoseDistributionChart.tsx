import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { LearningStep } from '../types';

interface Props {
  steps: LearningStep[];
  finalDose: number | null;
}

export function RecommendedDoseDistributionChart({ steps, finalDose }: Props) {
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
        type: 'category',
        name: '离焦剂量 (D)',
        nameLocation: 'middle',
        nameGap: 30,
        data: ['3.5-3.8', '3.8-4.1', '4.1-4.4', '4.4-4.7', '4.7-5.0'],
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
      },
      yAxis: {
        type: 'value',
        name: '频次',
        nameLocation: 'middle',
        nameGap: 35,
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#555' },
        splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } },
      },
      series: [
        {
          name: '推荐频次',
          type: 'bar',
          barWidth: '60%',
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#3498db' },
              { offset: 1, color: '#2980b9' },
            ]),
            borderRadius: [4, 4, 0, 0],
          },
          data: [],
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { type: 'dashed', width: 2, color: '#e74c3c' },
            label: { formatter: '推荐值', position: 'end' },
            data: [],
          },
        },
      ],
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params[0]) return '';
          const p = params[0];
          return `区间: ${p.name} D<br/>频次: ${p.value}`;
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

    const bins = [0, 0, 0, 0, 0];
    steps.forEach((s) => {
      if (s.dose >= 3.5 && s.dose < 3.8) bins[0]++;
      else if (s.dose >= 3.8 && s.dose < 4.1) bins[1]++;
      else if (s.dose >= 4.1 && s.dose < 4.4) bins[2]++;
      else if (s.dose >= 4.4 && s.dose < 4.7) bins[3]++;
      else if (s.dose >= 4.7 && s.dose <= 5.0) bins[4]++;
    });

    chartInstance.current.setOption({
      series: [
        {
          name: '推荐频次',
          data: bins,
          markLine: {
            data: finalDose !== null ? [{ xAxis: Math.floor((finalDose - 3.5) / 0.3) }] : [],
          },
        },
      ],
    });
  }, [steps, finalDose]);

  return (
    <div style={{
      background: 'white',
      borderRadius: '1rem',
      padding: '1rem',
      boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
      height: '100%',
    }}>
      <h3 style={{ margin: '0 0 0.5rem 0', color: '#1e3a5f', fontSize: '1rem' }}>
        推荐剂量分布
      </h3>
      <p style={{ margin: '0 0 1rem 0', color: '#777', fontSize: '0.8rem' }}>
        历史推荐剂量直方图
      </p>
      <div ref={chartRef} style={{ width: '100%', height: '280px' }} />
    </div>
  );
}